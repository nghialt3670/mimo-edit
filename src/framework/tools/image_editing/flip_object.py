from __future__ import annotations

from PIL import Image

from framework.inference.object_clear import object_clear_remove_object

from ..context import ToolContext
from ..registry import register_tool
from ..utils import (
    masked_layer,
    paste_layer,
    validate_image,
    validate_mask,
    validate_text,
)


def _validate_flip_object_args(
    image: Image.Image,
    mask: Image.Image,
    axis: str,
) -> None:
    validate_image("image", image)
    validate_mask(image, mask)
    validate_text("axis", axis)
    if axis not in ("x", "y"):
        raise ValueError("axis must be 'x' or 'y'")


@register_tool(
    "flip_object",
    "Flip the selected object horizontally or vertically.",
    {
        "image": {"type": "image", "description": "Image to edit."},
        "mask": {"type": "image", "description": "Mask selecting the object."},
        "axis": {
            "type": "string",
            "description": "'x' for horizontal; 'y' for vertical.",
        },
    },
    "image",
)
def flip_object(
    ctx: ToolContext,
    image: Image.Image,
    mask: Image.Image,
    axis: str,
) -> Image.Image:
    _validate_flip_object_args(image, mask, axis)
    transpose = {
        "x": Image.Transpose.FLIP_LEFT_RIGHT,
        "y": Image.Transpose.FLIP_TOP_BOTTOM,
    }
    layer, origin = masked_layer(image, mask)
    clean = object_clear_remove_object(ctx.model_pool, image, mask)
    return paste_layer(clean, layer.transpose(transpose[axis]), origin)
