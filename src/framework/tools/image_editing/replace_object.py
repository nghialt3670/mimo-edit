from __future__ import annotations

from PIL import Image

from framework.inference.object_clear import object_clear_remove_object

from ..context import ToolContext
from ..registry import register_tool
from ..utils import paste_layer, validate_image, validate_mask


def _validate_replace_object_args(
    image: Image.Image,
    mask: Image.Image,
    replacement: Image.Image,
) -> None:
    validate_image("image", image)
    validate_mask(image, mask)
    validate_image("replacement", replacement)


@register_tool(
    "replace_object",
    "Replace the selected object with another image.",
    {
        "image": {"type": "image", "description": "Image to edit."},
        "mask": {
            "type": "image",
            "description": "Mask selecting the object to replace.",
        },
        "replacement": {
            "type": "image",
            "description": "Image to place in the selected area.",
        },
    },
    "image",
)
def replace_object(
    ctx: ToolContext,
    image: Image.Image,
    mask: Image.Image,
    replacement: Image.Image,
) -> Image.Image:
    _validate_replace_object_args(image, mask, replacement)
    clean = object_clear_remove_object(ctx.model_pool, image, mask)
    bounds = mask.convert("L").resize(image.size, Image.Resampling.NEAREST).getbbox()
    if bounds is None:
        raise ValueError("mask is empty")
    fitted = replacement.convert("RGBA").resize(
        (bounds[2] - bounds[0], bounds[3] - bounds[1]), Image.Resampling.LANCZOS
    )
    return paste_layer(clean, fitted, (bounds[0], bounds[1]))
