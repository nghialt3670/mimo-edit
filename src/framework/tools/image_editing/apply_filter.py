from __future__ import annotations

from PIL import Image, ImageEnhance, ImageFilter, ImageOps

from ..context import ToolContext
from ..registry import register_tool
from ..utils import (
    composite_masked,
    validate_image,
    validate_mask,
    validate_number,
    validate_number_range,
    validate_text,
)


def _validate_apply_filter_args(
    image: Image.Image,
    filter_name: str,
    value: float,
    mask: Image.Image | None,
) -> float:
    validate_image("image", image)
    if mask is not None:
        validate_mask(image, mask)
    validate_text("filter_name", filter_name)
    _validate_filter_name(filter_name)
    value = validate_number("value", value)
    validate_number_range("value", value, -1, 1)
    return value


def _validate_filter_name(filter_name: str) -> None:
    if filter_name not in {
        "brightness",
        "contrast",
        "saturation",
        "blackWhite",
        "invert",
        "blur",
    }:
        raise ValueError(f"unsupported filter: {filter_name}")


@register_tool(
    "apply_filter",
    "Apply a filter to the image or blend it into a masked area.",
    {
        "image": {"type": "image", "description": "Image to edit."},
        "filter_name": {
            "type": "string",
            "description": "Filter: brightness, contrast, saturation, blackWhite, invert, or blur.",
        },
        "value": {"type": "number", "description": "Filter strength from -1 to 1."},
        "mask": {
            "type": "image",
            "description": "Optional mask for the area to filter; omit to filter the whole image.",
            "nullable": True,
        },
    },
    "image",
)
def apply_filter(
    ctx: ToolContext,
    image: Image.Image,
    filter_name: str,
    value: float,
    mask: Image.Image | None = None,
) -> Image.Image:
    value = _validate_apply_filter_args(image, filter_name, value, mask)
    if filter_name == "blackWhite":
        filtered = ImageOps.grayscale(image).convert("RGB")
    elif filter_name == "invert":
        filtered = ImageOps.invert(image.convert("RGB"))
    elif filter_name == "blur":
        filtered = image.filter(
            ImageFilter.GaussianBlur(radius=max(0, (value + 1) * 5))
        )
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
