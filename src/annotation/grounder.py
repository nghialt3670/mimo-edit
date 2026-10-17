from __future__ import annotations

from typing import Protocol

import numpy as np
from PIL import Image
from scipy import ndimage

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

    if best is None:
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


def _spatial_split(binary: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    ys, xs = np.nonzero(binary)
    if len(xs) < 2:
        return None

    points = np.column_stack((xs, ys)).astype(np.float64)
    center = points.mean(axis=0)
    first_index = int(np.argmax(np.sum((points - center) ** 2, axis=1)))
    second_index = int(np.argmax(np.sum((points - points[first_index]) ** 2, axis=1)))
    if first_index == second_index:
        return None

    centers = points[[first_index, second_index]]
    labels = np.zeros(len(points), dtype=np.int8)
    for _ in range(50):
        first_distances = np.sum((points - centers[0]) ** 2, axis=1)
        second_distances = np.sum((points - centers[1]) ** 2, axis=1)
        updated_labels = (second_distances < first_distances).astype(np.int8)
        if np.all(updated_labels == updated_labels[0]):
            return None
        updated_centers = np.vstack(
            [points[updated_labels == label].mean(axis=0) for label in (0, 1)]
        )
        if np.array_equal(updated_labels, labels):
            labels = updated_labels
            break
        labels = updated_labels
        centers = updated_centers

    first = np.zeros(binary.shape, dtype=bool)
    second = np.zeros(binary.shape, dtype=bool)
    first[ys[labels == 0], xs[labels == 0]] = True
    second[ys[labels == 1], xs[labels == 1]] = True
    if not first.any() or not second.any():
        return None
    return first, second


def _watershed_split(binary: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    if not binary.any():
        return None

    distance = ndimage.distance_transform_edt(binary)
    smoothed = ndimage.gaussian_filter(distance, sigma=2.0)
    peak_spacing = max(5, int(min(binary.shape) * 0.04))
    local_maxima = (
        (smoothed == ndimage.maximum_filter(smoothed, size=2 * peak_spacing + 1))
        & binary
        & (smoothed >= float(smoothed.max()) * 0.18)
    )
    peak_labels, peak_count = ndimage.label(local_maxima)

    peaks: list[tuple[float, int, int]] = []
    for label in range(1, peak_count + 1):
        ys, xs = np.nonzero(peak_labels == label)
        if not len(xs):
            continue
        best_index = int(np.argmax(smoothed[ys, xs]))
        peaks.append(
            (
                float(smoothed[ys[best_index], xs[best_index]]),
                int(ys[best_index]),
                int(xs[best_index]),
            )
        )
    if len(peaks) < 2:
        return None

    peak_pair = max(
        (
            (first, second)
            for index, first in enumerate(peaks)
            for second in peaks[index + 1 :]
        ),
        key=lambda pair: min(pair[0][0], pair[1][0])
        * np.hypot(pair[0][1] - pair[1][1], pair[0][2] - pair[1][2]),
    )
    if np.hypot(
        peak_pair[0][1] - peak_pair[1][1],
        peak_pair[0][2] - peak_pair[1][2],
    ) < max(6, int(min(binary.shape) * 0.04)):
        return None

    markers = np.zeros(binary.shape, dtype=np.int32)
    for label, (_, y, x) in enumerate(peak_pair, start=1):
        markers[y, x] = label

    interior_distance = smoothed[binary]
    topography = np.full(binary.shape, 255, dtype=np.uint8)
    topography[binary] = np.clip(
        255.0 * (1.0 - smoothed[binary] / float(interior_distance.max())),
        0,
        255,
    ).astype(np.uint8)
    labels = ndimage.watershed_ift(topography, markers)
    parts = tuple(binary & (labels == label) for label in (1, 2))
    if not parts[0].any() or not parts[1].any():
        return None
    return parts[0], parts[1]


def _balanced_split(
    split: tuple[np.ndarray, np.ndarray] | None,
    total_area: int,
) -> bool:
    if split is None or total_area == 0:
        return False
    smaller_area = min(int(part.sum()) for part in split)
    return smaller_area >= total_area * 0.12


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

    if expected_count is not None:
        while len(components) < expected_count:
            if not components:
                break
            split_index = max(
                range(len(components)), key=lambda index: components[index][0]
            )
            component_mask = np.asarray(components[split_index][1]) > 0
            component_area = int(component_mask.sum())
            split_candidates = (
                _projection_split(component_mask),
                _watershed_split(component_mask),
                _spatial_split(component_mask),
            )
            split = next(
                (
                    candidate
                    for candidate in split_candidates
                    if _balanced_split(candidate, component_area)
                ),
                None,
            )
            if split is None:
                break
            first, second = split
            if min(int(first.sum()), int(second.sum())) < 1:
                break
            components.pop(split_index)
            components.extend(
                (
                    (int(part.sum()), Image.fromarray(part.astype(np.uint8) * 255))
                    for part in (first, second)
                )
            )

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
