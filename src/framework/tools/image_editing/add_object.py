"""Registered tool for inserting one object at a box or mask location."""

from __future__ import annotations

from PIL import Image

from framework.inference import (
    flux2_klein_add_object_with_box,
    flux2_klein_add_object_with_mask,
    mice_add_object_with_box,
)

from ..context import ToolContext
from ..registry import register_tool
from ..utils import (
    validate_image,
    validate_mask,
    validate_number_list,
    validate_text,
)


def _validate_add_object_args(
    image: Image.Image,
    prompt: str,
    mask: Image.Image | None,
    box: list[float] | None,
) -> None:
    """Validate the shared inputs and the chosen placement method."""
    validate_image("image", image)
    validate_text("prompt", prompt)
    if (mask is None) == (box is None):
        raise ValueError("Provide exactly one of mask or box")

    if mask is not None:
        validate_mask(image, mask, same_size=True)
        return

    coordinates = validate_number_list("box", box, lengths=(4,))
    x1, y1, x2, y2 = coordinates
    if not (0 <= x1 < x2 <= image.width and 0 <= y1 < y2 <= image.height):
        raise ValueError("box must be a valid XYXY rectangle inside the image")


@register_tool(
    "add_object",
    "Add an object to the image. Provide exactly one placement input: box or mask. "
    "Say in anchor what the object goes on or next to (for example 'on the blue cooler').",
    {
        "image": {
            "type": "image",
            "description": "Image to edit.",
        },
        "prompt": {
            "type": "string",
            "description": "Describe the object to add, including appearance and orientation. Do not describe where it goes; use anchor for that.",
        },
        "box": {
            "type": "array",
            "description": "Approximate placement box [x1, y1, x2, y2] in image pixels.",
            "nullable": True,
        },
        "mask": {
            "type": "image",
            "description": "Mask the same size as the image; white marks where to add the object.",
            "nullable": True,
        },
    },
    "image",
)
def add_object(
    ctx: ToolContext,
    image: Image.Image,
    prompt: str,
    mask: Image.Image | None = None,
    box: list[float] | None = None,
) -> Image.Image:
    _validate_add_object_args(image, prompt, mask, box)

    if box is not None:
        return flux2_klein_add_object_with_box(
            ctx.model_pool, image, box, prompt
        )

    return flux2_klein_add_object_with_mask(
        ctx.model_pool, image, mask, prompt
    )
