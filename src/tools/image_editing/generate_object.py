from __future__ import annotations

from PIL import Image, ImageDraw

from utils.image import validate_box
from inference import (
    ModelCache,
    inpaint_with_gligen,
    inpaint_with_stable_diffusion,
)
from ..registry import register_tool


@register_tool(
    "generate_object",
    "Fill or generate content in one selected region. Provide one mask or one box and describe the desired result in prompt. For phrase-controlled object placement, provide a box and phrase; do not provide a mask in this mode.",
    {
        "image": {"type": "image", "description": "Source image."},
        "prompt": {"type": "string", "description": "Describe the content to generate or how to fill the selected region."},
        "mask": {
            "type": "image",
            "description": "Optional Pillow mask; white pixels mark the one region to fill. Pass null when using a box.",
            "nullable": True,
        },
        "box": {
            "type": "array",
            "description": "Optional pixel box [x1, y1, x2, y2] for the one region. Pair it with phrase to use GLIGEN.",
            "nullable": True,
        },
        "phrase": {
            "type": "string",
            "description": "Optional description of the one object to place inside box; when provided, use GLIGEN.",
            "nullable": True,
        },
    },
    "image",
)
def generate_object(
    models: ModelCache,
    image: Image.Image,
    prompt: str,
    mask: Image.Image | None = None,
    box: list[float] | None = None,
    phrase: str | None = None,
) -> Image.Image:
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("prompt must be a non-empty string")
    if phrase is not None:
        if not phrase.strip():
            raise ValueError("phrase must be a non-empty string")
        if mask is not None:
            raise ValueError("GLIGEN uses a box and phrase, not a mask")
        if box is None:
            raise ValueError("GLIGEN phrase requires a box")
        return inpaint_with_gligen(
            models,
            image,
            prompt,
            [phrase],
            [box],
            model_key="generate_object_gligen",
        )

    if mask is not None and box is not None:
        raise ValueError("Provide a mask or a box, not both")
    if mask is None and box is None:
        raise ValueError("Provide one mask or one box to select a generation region")
    if mask is None:
        mask = Image.new("L", image.size, 0)
        ImageDraw.Draw(mask).rectangle(validate_box(box), fill=255)
    return inpaint_with_stable_diffusion(
        models,
        image,
        mask,
        prompt,
        model_key="generate_object",
    )