from __future__ import annotations

from typing import Any

from PIL import Image, ImageEnhance, ImageFilter, ImageOps

from utils.image import composite_masked
from ..registry import register_tool


@register_tool(
    "apply_filter",
    "Apply a Pillow filter to the whole image or one masked object. With a mask, the filter is softly blended at the selected object's edge.",
    {
        "image": {"type": "image", "description": "Source image."},
        "filter_name": {"type": "string", "description": "Filter name."},
        "value": {"type": "number", "description": "Strength from -1 to 1; use 0 for defaults."},
        "mask": {
            "type": "image",
            "description": "Optional mask selecting one object. Omit or pass null to filter the whole image.",
            "nullable": True,
        },
    },
    "image",
)
def apply_filter(
    _models: Any,
    image: Image.Image,
    filter_name: str,
    value: float,
    mask: Image.Image | None = None,
) -> Image.Image:
    value = float(value)
    if not -1 <= value <= 1:
        raise ValueError("value must be between -1 and 1")
    if filter_name == "blackWhite":
        filtered = ImageOps.grayscale(image).convert("RGB")
    elif filter_name == "invert":
        filtered = ImageOps.invert(image.convert("RGB"))
    elif filter_name == "blur":
        filtered = image.filter(ImageFilter.GaussianBlur(radius=max(0, (value + 1) * 5)))
    else:
        enhancers = {
            "brightness": ImageEnhance.Brightness,
            "contrast": ImageEnhance.Contrast,
            "saturation": ImageEnhance.Color,
        }
        if filter_name not in enhancers:
            raise ValueError(f"unsupported filter: {filter_name}")
        filtered = enhancers[filter_name](image).enhance(1 + value)

    if mask is not None:
        return composite_masked(image, filtered, mask)
    return filtered