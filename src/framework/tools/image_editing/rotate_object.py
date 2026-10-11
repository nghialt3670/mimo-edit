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
    validate_number,
)


def _validate_rotate_object_args(
    image: Image.Image,
    mask: Image.Image,
    angle: float,
) -> None:
    validate_image("image", image)
    validate_mask(image, mask)
    validate_number("angle", angle)


@register_tool(
    "rotate_object",
    "Rotate the selected object counterclockwise.",
    {
        "image": {"type": "image", "description": "Image to edit."},
        "mask": {"type": "image", "description": "Mask selecting the object."},
        "angle": {"type": "number", "description": "Rotation angle in degrees."},
    },
    "image",
)
def rotate_object(
    ctx: ToolContext,
    image: Image.Image,
    mask: Image.Image,
    angle: float,
) -> Image.Image:
    _validate_rotate_object_args(image, mask, angle)
    layer, origin = masked_layer(image, mask)
    rotated = layer.rotate(float(angle), expand=True, resample=Image.Resampling.BICUBIC)
    center = (origin[0] + layer.width / 2, origin[1] + layer.height / 2)
    destination = (
        round(center[0] - rotated.width / 2),
        round(center[1] - rotated.height / 2),
    )
    clean = object_clear_remove_object(ctx.model_pool, image, mask)
    return paste_layer(clean, rotated, destination)
