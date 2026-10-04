from __future__ import annotations

from PIL import Image

from inference import ModelCache, scale_masked_object
from ..registry import register_tool


@register_tool(
    "scale_object",
    "Scale one masked object around its center.",
    {
        "image": {"type": "image", "description": "Source image."},
        "mask": {"type": "image", "description": "One Pillow mask selecting the object."},
        "factor": {"type": "number", "description": "Scale factor greater than zero; values above 1 enlarge the object."},
    },
    "image",
)
def scale_object(
    models: ModelCache,
    image: Image.Image,
    mask: Image.Image,
    factor: float,
) -> Image.Image:
    if factor <= 0:
        raise ValueError("factor must be greater than zero")
    return scale_masked_object(models, image, mask, factor)