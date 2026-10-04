from __future__ import annotations

from PIL import Image

from inference import ModelCache, edit_masked_object
from inference.editing import EditMode
from ..registry import register_tool


@register_tool(
    "edit_object",
    "Edit one masked object. Choose mode='local' to change appearance while keeping its shape; 'regional' to allow shape/accessory changes within nearby padding; or 'global' to give the model the whole image and allow scene-wide changes. Use regional for edits such as adding a hat. Padding is a 0-1 fraction of the object's bounding-box width and height on each side; regional defaults to 0.25.",
    {
        "image": {"type": "image", "description": "Source image."},
        "mask": {"type": "image", "description": "One Pillow mask; white pixels select the object."},
        "prompt": {"type": "string", "description": "Describe the change to make to the selected object."},
        "mode": {
            "type": "string",
            "description": "Required: 'local' preserves the selected object's silhouette; 'regional' permits changes/additions into a padded area around it; 'global' edits using the whole image and can change other scene areas.",
        },
        "padding": {
            "type": "number",
            "description": "Optional regional-mode padding from 0 to 1, as a fraction of the object's bounding-box dimensions per side. Default 0.25. Larger values allow larger nearby additions but can affect more surroundings.",
            "nullable": True,
        },
    },
    "image",
)
def edit_object(
    models: ModelCache,
    image: Image.Image,
    mask: Image.Image,
    prompt: str,
    mode: EditMode,
    padding: float | None = None,
) -> Image.Image:
    if not isinstance(image, Image.Image):
        raise TypeError("image must be a Pillow image")
    return edit_masked_object(
        models,
        image,
        mask,
        prompt,
        mode=mode,
        padding=0.25 if padding is None else padding,
    )