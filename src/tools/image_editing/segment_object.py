from __future__ import annotations

from PIL import Image

from inference import ModelCache, segment_mask, segment_text
from ..registry import register_tool


@register_tool(
    "segment_object",
    "Find one object mask using exactly one method: a text prompt, point, box, or scribble. Use a text prompt to find the highest-scoring matching object; use point, box, or scribble to select a specific region.",
    {
        "image": {"type": "image", "description": "Source image."},
        "prompt": {
            "type": "string",
            "description": "Optional text description of one object to find. Use this instead of point, box, or scribble.",
            "nullable": True,
        },
        "point": {
            "type": "array",
            "description": "Optional pixel point [x, y] selecting a specific object. Pass null when unused.",
            "nullable": True,
        },
        "box": {
            "type": "array",
            "description": "Optional pixel box [x1, y1, x2, y2] selecting a specific object. Pass null when unused.",
            "nullable": True,
        },
        "scribble": {
            "type": "array",
            "description": "Optional pixel path [[x, y], ...] drawn over a specific object. Pass null when unused.",
            "nullable": True,
        },
    },
    "image",
)
def segment_object(
    models: ModelCache,
    image: Image.Image,
    prompt: str | None = None,
    point: list[float] | None = None,
    box: list[float] | None = None,
    scribble: list[list[float]] | None = None,
) -> Image.Image:
    if prompt is not None and (not isinstance(prompt, str) or not prompt.strip()):
        raise ValueError("prompt must be a non-empty string")
    selected = sum(
        selector is not None for selector in (prompt, point, box, scribble)
    )
    if selected != 1:
        raise ValueError("Provide exactly one of prompt, point, box, or scribble")
    if prompt is not None:
        masks = segment_text(
            models, image, prompt, model_key="segment_object"
        )
        if not masks:
            raise ValueError(f"No object matched the text prompt: {prompt!r}")
        return masks[0]
    if point is not None:
        if len(point) != 2:
            raise ValueError("point must be [x, y]")
        x, y = (round(float(value)) for value in point)
        return segment_mask(
            models,
            image,
            points=[{"x": x, "y": y, "label": 1}],
            model_key="segment_object",
        )
    if box is not None:
        return segment_mask(models, image, box=box, model_key="segment_object")
    if not scribble:
        raise ValueError("scribble must contain at least one point")
    stride = max(1, len(scribble) // 10)
    points = [
        {"x": round(float(path_point[0])), "y": round(float(path_point[1])), "label": 1}
        for path_point in scribble[::stride][:10]
    ]
    return segment_mask(models, image, points=points, model_key="segment_object")