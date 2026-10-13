from __future__ import annotations

from PIL import Image

from .pool import ModelPool
from .sam3 import sam3_segment_mask


def grounding_dino_segment_region(
    pool: ModelPool,
    image: Image.Image,
    prompt: str,
) -> Image.Image:
    """Ground a text description and return a SAM 3 mask for its best box."""
    import torch

    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("prompt must be a non-empty string")

    source = image.convert("RGB")
    query = prompt.strip().rstrip(".")
    text_labels = [[query]]
    pipelines = pool.grounding_dino_pipelines
    inputs = pipelines.processor(
        images=source,
        text=text_labels,
        return_tensors="pt",
    ).to(pool.device)
    model_dtype = pipelines.model.dtype
    for name, value in inputs.items():
        if isinstance(value, torch.Tensor) and torch.is_floating_point(value):
            inputs[name] = value.to(dtype=model_dtype)
    with torch.inference_mode():
        outputs = pipelines.model(**inputs)
    result = pipelines.processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        threshold=pool.config.grounding_dino.box_threshold,
        text_threshold=pool.config.grounding_dino.text_threshold,
        target_sizes=[(source.height, source.width)],
        text_labels=text_labels,
    )[0]
    scores = result.get("scores")
    boxes = result.get("boxes")
    if scores is None or boxes is None or len(scores) == 0:
        raise ValueError(
            f"No image region matched the description: {prompt!r}. Try a concise "
            "description of a visible object or select it with a point, box, or "
            "scribble using segment_object."
        )

    best_index = int(scores.argmax().item())
    x0, y0, x1, y1 = (float(coordinate) for coordinate in boxes[best_index].tolist())
    x0 = min(max(x0, 0.0), float(source.width))
    y0 = min(max(y0, 0.0), float(source.height))
    x1 = min(max(x1, 0.0), float(source.width))
    y1 = min(max(y1, 0.0), float(source.height))
    if x1 <= x0 or y1 <= y0:
        raise ValueError(
            f"Grounding DINO returned an invalid region for description: {prompt!r}"
        )

    return sam3_segment_mask(pool, source, box=[x0, y0, x1, y1])


__all__ = ["grounding_dino_segment_region"]
