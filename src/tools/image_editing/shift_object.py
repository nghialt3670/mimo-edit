from __future__ import annotations

from PIL import Image

from utils.image import masked_layer, paste_layer
from inference import ModelCache, remove_with_object_clear
from ..registry import register_tool


@register_tool(
    "shift_object",
    "Shift one masked object by a pixel offset [dx, dy].",
    {
        "image": {"type": "image", "description": "Source image."},
        "mask": {"type": "image", "description": "One Pillow mask selecting the object."},
        "offset": {"type": "array", "description": "Pixel offset [dx, dy]."},
    },
    "image",
)
def shift_object(
    models: ModelCache,
    image: Image.Image,
    mask: Image.Image,
    offset: list[float],
) -> Image.Image:
    if len(offset) != 2:
        raise ValueError("offset must be [dx, dy]")
    layer, origin = masked_layer(image, mask)
    clean = remove_with_object_clear(
        models, image, mask, "remove the selected object", model_key="shift_object"
    )
    destination = (round(origin[0] + offset[0]), round(origin[1] + offset[1]))
    return paste_layer(clean, layer, destination)