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
    validate_positive_number,
)


def _validate_scale_object_args(
    image: Image.Image,
    mask: Image.Image,
    factor: float,
) -> None:
    validate_image("image", image)
    validate_mask(image, mask)
    validate_positive_number("factor", factor)


@register_tool(
    "scale_object",
    "Scale the selected object around its center.",
    {
        "image": {"type": "image", "description": "Image to edit."},
        "mask": {"type": "image", "description": "Mask selecting the object."},
        "factor": {
            "type": "number",
            "description": "Scale factor; above 1 enlarges and below 1 shrinks.",
        },
    },
    "image",
)
def scale_object(
    ctx: ToolContext,
    image: Image.Image,
    mask: Image.Image,
    factor: float,
) -> Image.Image:
    _validate_scale_object_args(image, mask, factor)
    layer, origin = masked_layer(image, mask)
    scaled = layer.resize(
        (max(1, round(layer.width * factor)), max(1, round(layer.height * factor))),
        Image.Resampling.LANCZOS,
    )
    center = (origin[0] + layer.width / 2, origin[1] + layer.height / 2)
    destination = (
        round(center[0] - scaled.width / 2),
        round(center[1] - scaled.height / 2),
    )
    clean = object_clear_remove_object(ctx.model_pool, image, mask)
    return paste_layer(clean, scaled, destination)
