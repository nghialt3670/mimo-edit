from __future__ import annotations

from PIL import Image

from inference import ModelCache, remove_with_object_clear
from ..registry import register_tool


@register_tool(
    "remove_object",
    "Remove one masked object using ObjectClear.",
    {
        "image": {"type": "image", "description": "Source image."},
        "mask": {"type": "image", "description": "One Pillow mask; white pixels select the object."},
    },
    "image",
)
def remove_object(
    models: ModelCache, image: Image.Image, mask: Image.Image
) -> Image.Image:
    return remove_with_object_clear(
        models, image, mask, "remove the selected object", model_key="remove_object"
    )