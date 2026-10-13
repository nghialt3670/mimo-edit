from __future__ import annotations

from PIL import Image

from framework.inference import grounding_dino_segment_region

from ..context import ToolContext
from ..registry import register_tool
from ..utils import validate_image, validate_text


@register_tool(
    "segment_region",
    "Find a region described in natural language and return a precise object mask.",
    {
        "image": {
            "type": "image",
            "description": "Image containing the region.",
        },
        "prompt": {
            "type": "string",
            "description": (
                "Natural-language description of the object or region to segment."
            ),
        },
    },
    "image",
)
def segment_region(
    ctx: ToolContext,
    image: Image.Image,
    prompt: str,
) -> Image.Image:
    validate_image("image", image)
    validate_text("prompt", prompt)
    return grounding_dino_segment_region(ctx.model_pool, image, prompt)
