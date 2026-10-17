from __future__ import annotations

from typing import Literal

from PIL import Image

from framework.inference import flux2_klein_edit_object

from ..context import ToolContext
from ..registry import register_tool
from ..utils import (
    validate_image,
    validate_mask,
    validate_number,
    validate_number_range,
    validate_text,
)

BOX_EXPAND_RATIO = 0.1


def _expanded_box_mask(
    mask: Image.Image, ratio: float = BOX_EXPAND_RATIO
) -> Image.Image:
    left, top, right, bottom = mask.convert("L").getbbox()
    pad_x = round((right - left) * ratio)
    pad_y = round((bottom - top) * ratio)
    box_mask = Image.new("L", mask.size, 0)
    box_mask.paste(
        255,
        (
            max(0, left - pad_x),
            max(0, top - pad_y),
            min(mask.width, right + pad_x),
            min(mask.height, bottom + pad_y),
        ),
    )
    return box_mask


def _validate_edit_object_args(
    image: Image.Image,
    mask: Image.Image,
    prompt: str,
    mode: str,
    padding: float | None,
) -> None:
    validate_image("image", image)
    validate_mask(image, mask, same_size=True)
    validate_text("prompt", prompt)
    if mode not in ("local", "regional", "global"):
        raise ValueError("mode must be 'local', 'regional', or 'global'")
    if padding is not None:
        validate_number_range("padding", validate_number("padding", padding), 0, 1)


@register_tool(
    "edit_object",
    "Edit the selected object. Describe the desired change and choose how much of the surrounding image may change.",
    {
        "image": {"type": "image", "description": "Image to edit."},
        "mask": {"type": "image", "description": "Mask selecting the object to edit."},
        "prompt": {
            "type": "string",
            "description": "Describe the intended change and preserve details the user did not ask to change.",
        },
        "mode": {
            "type": "string",
            "description": "'local' keeps the object's shape; 'regional' allows nearby changes; 'global' allows changes across the image.",
        },
        "padding": {
            "type": "number",
            "description": "Optional regional padding from 0 to 1. Defaults to 0.25.",
            "nullable": True,
        },
    },
    "image",
)
def edit_object(
    ctx: ToolContext,
    image: Image.Image,
    mask: Image.Image,
    prompt: str,
    mode: Literal["local", "regional", "global"],
    padding: float | None = None,
) -> Image.Image:
    if padding is None:
        padding = 0.25
    _validate_edit_object_args(image, mask, prompt, mode, padding)
    return flux2_klein_edit_object(
        ctx.model_pool,
        image,
        _expanded_box_mask(mask),
        prompt,
        mode=mode,
        padding=padding,
    )
