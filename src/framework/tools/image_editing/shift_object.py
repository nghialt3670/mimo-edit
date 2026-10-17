from __future__ import annotations

from PIL import Image

from framework.inference.flux2_klein import flux2_klein_remove_object

from ..context import ToolContext
from ..registry import register_tool
from ..utils import (
    masked_layer,
    paste_layer,
    validate_image,
    validate_mask,
    validate_number_list,
)


def _validate_shift_object_args(
    image: Image.Image,
    mask: Image.Image,
    offset: list[float],
) -> None:
    validate_image("image", image)
    validate_mask(image, mask)
    validate_number_list("offset", offset, lengths=(2,))


@register_tool(
    "shift_object",
    "Shift the selected object by a pixel offset.",
    {
        "image": {"type": "image", "description": "Image to edit."},
        "mask": {"type": "image", "description": "Mask selecting the object."},
        "offset": {"type": "array", "description": "Pixel offset [dx, dy]."},
    },
    "image",
)
def shift_object(
    ctx: ToolContext,
    image: Image.Image,
    mask: Image.Image,
    offset: list[float],
) -> Image.Image:
    _validate_shift_object_args(image, mask, offset)
    layer, origin = masked_layer(image, mask)
    clean = flux2_klein_remove_object(ctx.model_pool, image, mask)
    destination = (round(origin[0] + offset[0]), round(origin[1] + offset[1]))
    return paste_layer(clean, layer, destination)
