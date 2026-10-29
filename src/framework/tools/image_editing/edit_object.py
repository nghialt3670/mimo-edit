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


def _validate_edit_object_args(
    image: Image.Image,
    mask: Image.Image,
    prompt: str,
    mode: str,
    padding: float | None,
    reference_image: Image.Image | None = None,
    reference_mask: Image.Image | None = None,
) -> None:
    validate_image("image", image)
    validate_mask(image, mask, same_size=True)
    validate_text("prompt", prompt)
    if mode not in ("local", "regional", "global"):
        raise ValueError("mode must be 'local', 'regional', or 'global'")
    if padding is not None:
        validate_number_range("padding", validate_number("padding", padding), 0, 1)
    reference_source = image
    if reference_image is not None:
        reference_source = validate_image("reference_image", reference_image)
    if reference_mask is not None:
        validate_mask(
            reference_source,
            reference_mask,
            name="reference_mask",
            same_size=True,
        )


@register_tool(
    "edit_object",
    "Edit the selected object. Describe the desired change and choose how much of the surrounding image may change. To match color, material or style from another object, pass it as reference_image and reference_mask.",
    {
        "image": {"type": "image", "description": "Image to edit."},
        "mask": {"type": "image", "description": "Mask selecting the object to edit."},
        "prompt": {
            "type": "string",
            "description": "Describe the intended change and preserve details the user did not ask to change. For color changes say to keep the texture, material and lighting.",
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
        "reference_image": {
            "type": "image",
            "description": "Optional image containing the appearance to transfer (color, material, texture or style). Defaults to the source image.",
            "nullable": True,
        },
        "reference_mask": {
            "type": "image",
            "description": "Optional mask selecting the reference object in reference_image. Must match its dimensions.",
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
    reference_image: Image.Image | None = None,
    reference_mask: Image.Image | None = None,
) -> Image.Image:
    if padding is None:
        padding = 0.25
    _validate_edit_object_args(
        image, mask, prompt, mode, padding, reference_image, reference_mask
    )
    return flux2_klein_edit_object(
        ctx.model_pool,
        image,
        mask,
        prompt,
        mode=mode,
        padding=padding,
        reference_image=reference_image,
        reference_mask=reference_mask,
    )
