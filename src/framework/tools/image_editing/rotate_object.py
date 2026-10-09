from __future__ import annotations

from PIL import Image

from framework.utils.image import masked_layer, paste_layer
from framework.inference import ModelCache, remove_with_object_clear
from ..registry import register_tool


@register_tool(
    "rotate_object",
    "Rotate one masked object counterclockwise by an angle in degrees.",
    {
        "image": {"type": "image", "description": "Source image."},
        "mask": {"type": "image", "description": "One Pillow mask selecting the object."},
        "angle": {"type": "number", "description": "Counterclockwise rotation angle in degrees."},
    },
    "image",
)
def rotate_object(
    models: ModelCache,
    image: Image.Image,
    mask: Image.Image,
    angle: float,
) -> Image.Image:
    layer, origin = masked_layer(image, mask)
    rotated = layer.rotate(float(angle), expand=True, resample=Image.Resampling.BICUBIC)
    center = (origin[0] + layer.width / 2, origin[1] + layer.height / 2)
    destination = (round(center[0] - rotated.width / 2), round(center[1] - rotated.height / 2))
    clean = remove_with_object_clear(
        models, image, mask, "remove the selected object", model_key="rotate_object"
    )
    return paste_layer(clean, rotated, destination)