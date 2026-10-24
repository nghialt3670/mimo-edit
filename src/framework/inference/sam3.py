from __future__ import annotations

import numpy as np
from PIL import Image

from .pool import ModelPool
from .utils import validate_box, validate_number_range


def _keep_prompt_components(
    binary: np.ndarray, points: list[dict[str, int]] | None
) -> np.ndarray:
    from scipy import ndimage

    labels, count = ndimage.label(binary)
    if count == 0:
        return binary
    keep = set()
    for point in points or []:
        if point.get("label", 1) != 1:
            continue
        x, y = point["x"], point["y"]
        if 0 <= y < binary.shape[0] and 0 <= x < binary.shape[1] and labels[y, x]:
            keep.add(int(labels[y, x]))
    if not keep:
        sizes = ndimage.sum_labels(binary, labels, index=range(1, count + 1))
        keep.add(int(np.argmax(sizes)) + 1)
    return np.isin(labels, list(keep)).astype("uint8")


def sam3_segment_mask(
    pool: ModelPool,
    image: Image.Image,
    *,
    points: list[dict[str, int]] | None = None,
    box: list[float] | None = None,
) -> Image.Image:
    import torch

    if not points and box is None:
        raise ValueError("Provide at least one point or a box")
    input_points = (
        [[[[point["x"], point["y"]] for point in points]]] if points else None
    )
    input_labels = [[[point.get("label", 1) for point in points]]] if points else None
    input_boxes = [[list(validate_box(box))]] if box is not None else None
    inputs = pool.sam3_pipelines.tracker_processor(
        images=image.convert("RGB"),
        input_points=input_points,
        input_labels=input_labels,
        input_boxes=input_boxes,
        return_tensors="pt",
    ).to(str(pool.sam3_pipelines.tracker_model.device))
    with torch.inference_mode():
        outputs = pool.sam3_pipelines.tracker_model(**inputs)
    scores = outputs.iou_scores[0, 0].cpu().tolist()
    best_index = max(range(len(scores)), key=scores.__getitem__)
    mask = pool.sam3_pipelines.tracker_processor.post_process_masks(
        outputs.pred_masks, original_sizes=inputs["original_sizes"]
    )[0][0][best_index]
    binary = (mask.cpu().numpy() > pool.config.sam3.mask_threshold).astype("uint8")
    binary = _keep_prompt_components(binary, points)
    return Image.fromarray(binary * 255, mode="L")


def sam3_segment_masks(
    pool: ModelPool,
    image: Image.Image,
    prompt: str,
    *,
    min_confidence: float = 0.0,
) -> list[Image.Image]:
    """Return text-prompt masks in score order, omitting low-confidence results."""
    import torch

    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("prompt must be a non-empty string")
    validate_number_range("min_confidence", min_confidence, 0, 1)
    inputs = pool.sam3_pipelines.text_processor(
        images=image.convert("RGB"), text=prompt, return_tensors="pt"
    ).to(str(pool.sam3_pipelines.text_model.device))
    with torch.inference_mode():
        outputs = pool.sam3_pipelines.text_model(**inputs)
    results = pool.sam3_pipelines.text_processor.post_process_instance_segmentation(
        outputs, target_sizes=inputs["original_sizes"].tolist()
    )[0]
    threshold = pool.config.sam3.mask_threshold
    ordered_results = sorted(
        zip(results.get("scores", []), results.get("masks", [])),
        key=lambda item: float(item[0]),
        reverse=True,
    )
    masks = []
    for confidence, mask in ordered_results:
        if float(confidence) < min_confidence:
            continue
        binary = (mask.cpu().numpy() > threshold).astype("uint8")
        if binary.any():
            masks.append(Image.fromarray(binary * 255, mode="L"))
    return masks


__all__ = ["sam3_segment_mask", "sam3_segment_masks"]
