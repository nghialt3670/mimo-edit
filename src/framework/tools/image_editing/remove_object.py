from __future__ import annotations

from PIL import Image

from framework.inference import object_clear_remove_object

from ..context import ToolContext
from ..registry import register_tool
from ..utils import validate_image, validate_mask


def _validate_remove_object_args(image: Image.Image, mask: Image.Image) -> None:
    validate_image("image", image)
    validate_mask(image, mask)


@register_tool(
    "remove_object",
    "Remove the selected object from the image.",
    {
        "image": {"type": "image", "description": "Image to edit."},
        "mask": {
            "type": "image",
            "description": "Mask selecting the object to remove.",
        },
    },
    "image",
)
def remove_object(
    ctx: ToolContext, image: Image.Image, mask: Image.Image
) -> Image.Image:
    _validate_remove_object_args(image, mask)
    return object_clear_remove_object(
        ctx.model_pool, image, mask, "remove the selected object"
    )
