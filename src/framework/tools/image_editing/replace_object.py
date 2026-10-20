from __future__ import annotations

from PIL import Image

from framework.inference import (
    flux2_klein_add_object_with_box,
    flux2_klein_remove_object,
)
from framework.inference.flux2_klein.placement import describe_anchor

from ..context import ToolContext
from ..registry import register_tool
from ..utils import paste_layer, validate_image, validate_mask, validate_text


def _validate_replace_object_args(
    image: Image.Image,
    mask: Image.Image,
    replacement: Image.Image | None,
    prompt: str | None,
) -> tuple[int, int, int, int]:
    validate_image("image", image)
    validate_mask(image, mask)
    if (replacement is None) == (prompt is None):
        raise ValueError("Provide exactly one of replacement or prompt")
    if replacement is not None:
        validate_image("replacement", replacement)
    if prompt is not None:
        validate_text("prompt", prompt)
    bounds = mask.convert("L").resize(image.size, Image.Resampling.NEAREST).getbbox()
    if bounds is None:
        raise ValueError("mask is empty")
    return bounds


@register_tool(
    "replace_object",
    "Replace the selected object either with a supplied replacement image or with an object described by a text prompt.",
    {
        "image": {"type": "image", "description": "Image to edit."},
        "mask": {
            "type": "image",
            "description": "Mask selecting the object to replace.",
        },
        "replacement": {
            "type": "image",
            "description": "Optional replacement image to fit into the selected object's bounds. Provide this or prompt, not both.",
            "nullable": True,
        },
        "prompt": {
            "type": "string",
            "description": "Optional description of the object Klein should add after the selected object is removed. Provide this or replacement, not both.",
            "nullable": True,
        },
        "anchor": {
            "type": "string",
            "description": "With prompt: short placement phrase for what the new object rests on, e.g. 'on the wooden deck'. Inferred from the image if omitted.",
            "nullable": True,
        },
    },
    "image",
)
def replace_object(
    ctx: ToolContext,
    image: Image.Image,
    mask: Image.Image,
    replacement: Image.Image | None = None,
    prompt: str | None = None,
    anchor: str | None = None,
) -> Image.Image:
    bounds = _validate_replace_object_args(image, mask, replacement, prompt)
    clean = flux2_klein_remove_object(ctx.model_pool, image, mask)
    if prompt is not None:
        if anchor is None:
            anchor = describe_anchor(clean, bounds, prompt)
        return flux2_klein_add_object_with_box(
            ctx.model_pool,
            clean,
            bounds,
            prompt,
            anchor=anchor,
        )

    if replacement is None:
        raise RuntimeError("replacement or prompt must be provided")
    fitted = replacement.convert("RGBA").resize(
        (bounds[2] - bounds[0], bounds[3] - bounds[1]), Image.Resampling.LANCZOS
    )
    return paste_layer(clean, fitted, (bounds[0], bounds[1]))
