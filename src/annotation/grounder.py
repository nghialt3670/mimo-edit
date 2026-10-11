from __future__ import annotations

from typing import Protocol

import numpy as np
from PIL import Image

from .schema import ParsedEntity, Region, Sample


class Grounder(Protocol):
    def ground(
        self, sample: Sample, entity: ParsedEntity, visible_target_count: int
    ) -> Region | None: ...


class MaskOnlyGrounder:
    """Reject any attempt to ground a reference without a dataset mask."""

    def ground(
        self, sample: Sample, entity: ParsedEntity, visible_target_count: int
    ) -> Region | None:
        _ = sample, visible_target_count
        raise RuntimeError(
            f"Mask-only annotation cannot ground {entity.grounding_query!r} without a dataset mask"
        )


def resize_mask(mask: Image.Image, size: tuple[int, int]) -> Image.Image:
    return (
        mask.convert("L").resize(size, Image.Resampling.NEAREST)
        if mask.size != size
        else mask.convert("L")
    )


def mask_bbox(
    mask: Image.Image, size: tuple[int, int]
) -> tuple[float, float, float, float] | None:
    arr = np.asarray(resize_mask(mask, size))
    ys, xs = np.where(arr > 0)
    if xs.size == 0:
        return None
    return float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)


def _projection_split(binary: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    area = int(binary.sum())
    if area == 0:
        return None

    best: tuple[float, int, int] | None = None
    for axis in (0, 1):
        projection = binary.sum(axis=1 - axis).astype(np.float32)
        window = max(5, min(31, len(projection) // 20))
        smoothed = np.convolve(
            projection,
            np.full(window, 1 / window, dtype=np.float32),
            mode="same",
        )
        cumulative = np.cumsum(projection)
        minimum_side = max(64, int(area * 0.12))
        valid = np.flatnonzero(
            (cumulative >= minimum_side) & (cumulative <= area - minimum_side)
        )
        neighborhood = max(12, len(projection) // 12)

        for cut in valid:
            if cut == 0 or cut + 1 == len(projection):
                continue
            left_peak = float(smoothed[max(0, cut - neighborhood) : cut].max(initial=0))
            right_peak = float(
                smoothed[cut + 1 : min(len(projection), cut + neighborhood + 1)].max(
                    initial=0
                )
            )
            reference_peak = min(left_peak, right_peak)
            if reference_peak <= 0:
                continue
            valley_ratio = float(smoothed[cut]) / reference_peak
            balance_penalty = 0.02 * abs(float(cumulative[cut]) / area - 0.5)
            score = valley_ratio + balance_penalty
            if best is None or score < best[0]:
                best = score, axis, int(cut)

    if best is None or best[0] > 0.85:
        return None

    _, axis, cut = best
    first = binary.copy()
    second = binary.copy()
    if axis == 0:
        first[cut + 1 :, :] = False
        second[: cut + 1, :] = False
    else:
        first[:, cut + 1 :] = False
        second[:, : cut + 1] = False
    return first, second


def mask_components(
    mask: Image.Image,
    expected_count: int | None = None,
) -> list[tuple[int, Image.Image]]:
    binary = np.asarray(mask.convert("L")) > 0
    visited = np.zeros(binary.shape, dtype=bool)
    height, width = binary.shape
    components: list[tuple[int, Image.Image]] = []

    for y, x in np.argwhere(binary):
        y, x = int(y), int(x)
        if visited[y, x]:
            continue

        stack = [(y, x)]
        visited[y, x] = True
        pixels: list[tuple[int, int]] = []
        while stack:
            current_y, current_x = stack.pop()
            pixels.append((current_y, current_x))
            for neighbor_y, neighbor_x in (
                (current_y - 1, current_x),
                (current_y + 1, current_x),
                (current_y, current_x - 1),
                (current_y, current_x + 1),
            ):
                if (
                    0 <= neighbor_y < height
                    and 0 <= neighbor_x < width
                    and binary[neighbor_y, neighbor_x]
                    and not visited[neighbor_y, neighbor_x]
                ):
                    visited[neighbor_y, neighbor_x] = True
                    stack.append((neighbor_y, neighbor_x))

        component = np.zeros(binary.shape, dtype=np.uint8)
        ys, xs = zip(*pixels, strict=True)
        component[ys, xs] = 255
        components.append((len(pixels), Image.fromarray(component)))

    if expected_count is not None and len(components) < expected_count:
        split = _projection_split(binary) if expected_count == 2 else None
        if split is not None:
            components = [
                (int(part.sum()), Image.fromarray(part.astype(np.uint8) * 255))
                for part in split
                if part.any()
            ]

    return sorted(components, key=lambda item: item[0], reverse=True)


class GroundingDinoGrounder:
    """Lazy Grounding DINO wrapper using Hugging Face Transformers."""

    def __init__(
        self,
        model_id: str = "IDEA-Research/grounding-dino-tiny",
        threshold: float = 0.35,
        text_threshold: float = 0.25,
        device: str | None = None,
    ):
        self.model_id = model_id
        self.threshold = threshold
        self.text_threshold = text_threshold
        self.device = device
        self._loaded = False

    def _load(self) -> None:
        if self._loaded:
            return
        import torch
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        if self.device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif (
                getattr(torch.backends, "mps", None)
                and torch.backends.mps.is_available()
            ):
                self.device = "mps"
            else:
                self.device = "cpu"
        self.torch = torch
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(
            self.model_id
        ).to(self.device)
        self.model.eval()
        self._loaded = True

    def ground(
        self, sample: Sample, entity: ParsedEntity, visible_target_count: int = 0
    ) -> Region | None:
        self._load()
        image = sample.image.convert("RGB")
        query = entity.grounding_query.strip()
        if not query:
            return None
        text_labels = [[query]]
        inputs = self.processor(images=image, text=text_labels, return_tensors="pt").to(
            self.device
        )
        with self.torch.inference_mode():
            outputs = self.model(**inputs)
        result = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=self.threshold,
            text_threshold=self.text_threshold,
            target_sizes=[(image.height, image.width)],
            text_labels=text_labels,
        )[0]
        scores, boxes = result.get("scores"), result.get("boxes")
        if scores is None or boxes is None or len(scores) == 0:
            return None
        i = int(scores.argmax().item())
        labels = result.get("text_labels")
        if labels is None:
            labels = result.get("labels")
        label = str(labels[i]) if labels is not None else query
        x0, y0, x1, y1 = (float(x) for x in boxes[i].tolist())
        x0 = min(max(x0, 0.0), float(image.width))
        y0 = min(max(y0, 0.0), float(image.height))
        x1 = min(max(x1, 0.0), float(image.width))
        y1 = min(max(y1, 0.0), float(image.height))
        return Region(
            bbox_xyxy=(x0, y0, x1, y1),
            image_size=image.size,
            confidence=float(scores[i].item()),
            label=label,
            source="grounding_dino",
        )


class HybridGrounder:
    """Use a benchmark mask for one visible target; Grounding DINO otherwise."""

    def __init__(
        self, fallback: Grounder | None = None, prefer_benchmark_mask: bool = True
    ):
        self.fallback = fallback or GroundingDinoGrounder()
        self.prefer_benchmark_mask = prefer_benchmark_mask

    def ground(
        self, sample: Sample, entity: ParsedEntity, visible_target_count: int
    ) -> Region | None:
        if sample.benchmark == "refedit_bench":
            if sample.visible_reference_count != 1 or sample.mask is None:
                return None
            bbox = mask_bbox(sample.mask, sample.image.size)
            if bbox is None:
                return None
            return Region(
                bbox_xyxy=bbox,
                image_size=sample.image.size,
                confidence=1.0,
                label=entity.grounding_query,
                source=f"benchmark_mask:{sample.metadata.get('mask_kind', 'object_instance')}",
            )

        is_single_target = entity.role == "target" and visible_target_count == 1
        if self.prefer_benchmark_mask and is_single_target and sample.mask is not None:
            bbox = mask_bbox(sample.mask, sample.image.size)
            if bbox is not None:
                return Region(
                    bbox_xyxy=bbox,
                    image_size=sample.image.size,
                    confidence=1.0,
                    label=entity.grounding_query,
                    source=f"benchmark_mask:{sample.metadata.get('mask_kind', 'region')}",
                )
        return self.fallback.ground(sample, entity, visible_target_count)
