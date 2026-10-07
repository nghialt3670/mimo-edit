from __future__ import annotations

from PIL import Image

from inference import ModelCache, edit_masked_object
from inference.editing import EditMode
from ..registry import register_tool


@register_tool(
    "edit_object",
    "Edit one masked object. Write prompt as a complete, actionable visual transformation, not just a label or category (for example, do not use only 'a woman'). State what to change and include the requested attributes; preserve unmentioned details and the surrounding scene unless the user asks to change them. Example: for 'Let the man with longer curly hair, wearing a loose tie be a woman,' use 'Transform the selected man into a woman, keeping his longer curly hair, loose tie, pose, and surrounding scene unchanged.' Choose mode='local' for appearance changes that keep the object's shape; 'regional' for identity/shape changes or additions within nearby padding; or 'global' only when the requested edit needs scene-wide changes. Padding is a 0-1 fraction of the object's bounding-box width and height on each side; regional defaults to 0.25.",
    {
        "image": {"type": "image", "description": "Source image."},
        "mask": {"type": "image", "description": "One Pillow mask; white pixels select the object."},
        "prompt": {
            "type": "string",
            "description": "A complete, actionable visual transformation: say what the selected object should look like after editing, include the user's requested attributes, and preserve unmentioned details. Do not pass only a subject/category label such as 'a woman'. Example: 'Transform the selected man into a woman, keeping his longer curly hair, loose tie, and pose.'",
        },
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