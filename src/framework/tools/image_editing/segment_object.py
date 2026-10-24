from __future__ import annotations

from PIL import Image

from framework.inference import (
    sam3_segment_mask,
    sam3_segment_masks,
)

from ..context import ToolContext
from ..registry import register_tool
from ..utils import validate_image, validate_number_list, validate_text


def _validate_segment_object_args(
    text: str | None,
    point: list[float] | None,
    box: list[float] | None,
    scribble: list[list[float]] | None,
    image: Image.Image,
) -> None:
    validate_image("image", image)
    selected = sum(selector is not None for selector in (text, point, box, scribble))
    if selected != 1:
        raise ValueError("Provide exactly one of text, point, box, or scribble")
    if text is not None:
        validate_text("text", text)
    if point is not None:
        _validate_segmentation_coordinates("point", point, image)
    if box is not None:
        x1, y1, x2, y2 = validate_number_list("box", box, lengths=(4,))
        if not (0 <= x1 < x2 <= image.width and 0 <= y1 < y2 <= image.height):
            raise ValueError("box must be a valid XYXY rectangle inside the image")
    if scribble is not None:
        if not scribble:
            raise ValueError("scribble must contain at least one point")
        for point in scribble:
            _validate_segmentation_coordinates("scribble point", point, image)


def _validate_segmentation_coordinates(
    name: str, coordinates: list[float], image: Image.Image
) -> None:
    numbers = validate_number_list(name, coordinates, lengths=(2,))
    x, y = numbers
    if not (0 <= x < image.width and 0 <= y < image.height):
        raise ValueError(f"{name} must be inside the image")


@register_tool(
    "segment_object",
    "Select an object by text, point, box, or scribble and return its mask.",
    {
        "image": {"type": "image", "description": "Image containing the object."},
        "text": {
            "type": "string",
            "description": "Description of the object to select. Use only one selection method.",
            "nullable": True,
        },
        "point": {
            "type": "array",
            "description": "Pixel point [x, y] on the object. Use only one selection method.",
            "nullable": True,
        },
        "box": {
            "type": "array",
            "description": "Pixel box [x1, y1, x2, y2] around the object. Use only one selection method.",
            "nullable": True,
        },
        "scribble": {
            "type": "array",
            "description": "Pixel path [[x, y], ...] drawn over the object. Use only one selection method.",
            "nullable": True,
        },
    },
    "image",
)
def segment_object(
    ctx: ToolContext,
    image: Image.Image,
    text: str | None = None,
    point: list[float] | None = None,
    box: list[float] | None = None,
    scribble: list[list[float]] | None = None,
) -> Image.Image:
    _validate_segment_object_args(text, point, box, scribble, image)
    if text is not None:
        masks = sam3_segment_masks(ctx.model_pool, image, text)
        if not masks:
            raise ValueError(
                f"No object matched the text description: {text!r}. Try a concise "
                "visible-object description, or locate the region with locate_region "
                "and pass its box, or select the object with a point or scribble."
            )
        return masks[0]
    if point is not None:
        x, y = (round(float(value)) for value in point)
        return sam3_segment_mask(
            ctx.model_pool,
            image,
            points=[{"x": x, "y": y, "label": 1}],
        )
    if box is not None:
        return sam3_segment_mask(ctx.model_pool, image, box=box)
    assert scribble is not None
    stride = max(1, len(scribble) // 10)
    points = [
        {"x": round(float(path_point[0])), "y": round(float(path_point[1])), "label": 1}
        for path_point in scribble[::stride][:10]
    ]
    return sam3_segment_mask(ctx.model_pool, image, points=points)
