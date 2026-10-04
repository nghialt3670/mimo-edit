from __future__ import annotations

from typing import Any

from PIL import Image

from .runtime import ModelCache, device, torch_dtype
from utils.image import validate_box


def _load_sam3(model_id: str) -> tuple[Any, Any, Any, Any]:
    from transformers import Sam3Model, Sam3Processor, Sam3TrackerModel, Sam3TrackerProcessor

    model_device = device()
    dtype = torch_dtype(model_device)
    return (
        Sam3Model.from_pretrained(model_id, torch_dtype=dtype).to(model_device).eval(),
        Sam3Processor.from_pretrained(model_id),
        Sam3TrackerModel.from_pretrained(model_id, torch_dtype=dtype).to(model_device).eval(),
        Sam3TrackerProcessor.from_pretrained(model_id),
    )


def segment_mask(
    models: ModelCache,
    image: Image.Image,
    *,
    points: list[dict[str, int]] | None = None,
    box: list[float] | None = None,
    model_key: str = "segment_object",
) -> Image.Image:
    import torch

    if not points and box is None:
        raise ValueError("Provide at least one point or a box")
    model_id = models.config.model_id(model_key, "facebook/sam3")
    _, _, tracker, processor = models.get(
        "sam3", lambda: _load_sam3(model_id), model_id=model_id
    )
    input_points = (
        [[[[point["x"], point["y"]] for point in points]]] if points else None
    )
    input_labels = (
        [[[point.get("label", 1) for point in points]]] if points else None
    )
    input_boxes = [[list(validate_box(box))]] if box is not None else None
    inputs = processor(
        images=image.convert("RGB"),
        input_points=input_points,
        input_labels=input_labels,
        input_boxes=input_boxes,
        return_tensors="pt",
    ).to(str(tracker.device))
    with torch.inference_mode():
        outputs = tracker(**inputs)
    scores = outputs.iou_scores[0, 0].cpu().tolist()
    best_index = max(range(len(scores)), key=scores.__getitem__)
    mask = processor.post_process_masks(
        outputs.pred_masks, original_sizes=inputs["original_sizes"]
    )[0][0][best_index]
    return Image.fromarray((mask.cpu().numpy() > 0).astype("uint8") * 255, mode="L")


def segment_text(
    models: ModelCache,
    image: Image.Image,
    prompt: str,
    *,
    model_key: str = "segment_object",
) -> list[Image.Image]:
    import torch

    model_id = models.config.model_id(model_key, "facebook/sam3")
    model, processor, _, _ = models.get(
        "sam3", lambda: _load_sam3(model_id), model_id=model_id
    )
    inputs = processor(
        images=image.convert("RGB"), text=prompt, return_tensors="pt"
    ).to(str(model.device))
    with torch.inference_mode():
        outputs = model(**inputs)
    results = processor.post_process_instance_segmentation(
        outputs, target_sizes=inputs["original_sizes"].tolist()
    )[0]
    ordered = sorted(
        zip(results.get("scores", []), results.get("masks", [])),
        key=lambda item: float(item[0]),
        reverse=True,
    )
    return [
        Image.fromarray((mask.cpu().numpy() > 0).astype("uint8") * 255, mode="L")
        for _, mask in ordered
    ]
