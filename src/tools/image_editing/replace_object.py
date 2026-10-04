from __future__ import annotations

from PIL import Image

from utils.image import paste_layer
from inference import ModelCache, remove_with_object_clear
from ..registry import register_tool


@register_tool(
    "replace_object",
    "Replace one masked object with one provided image. The replacement is resized to fit the mask bounds.",
    {
        "image": {"type": "image", "description": "Source image."},
        "mask": {"type": "image", "description": "One Pillow mask selecting the object to replace."},
        "replacement": {"type": "image", "description": "Pillow image to place in the selected object's bounds."},
    },
    "image",
)
def replace_object(
    models: ModelCache,
    image: Image.Image,
    mask: Image.Image,
    replacement: Image.Image,
) -> Image.Image:
    clean = remove_with_object_clear(
        models,
        image,
        mask,
        "remove the selected object",
        model_key="replace_object",
    )
    bounds = mask.convert("L").resize(image.size, Image.Resampling.NEAREST).getbbox()
    if bounds is None:
        raise ValueError("mask is empty")
    fitted = replacement.convert("RGBA").resize(
        (bounds[2] - bounds[0], bounds[3] - bounds[1]), Image.Resampling.LANCZOS
    )
    return paste_layer(clean, fitted, (bounds[0], bounds[1]))