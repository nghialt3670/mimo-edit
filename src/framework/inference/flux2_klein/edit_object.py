"""Edit an existing object with FLUX.2 Klein."""

from __future__ import annotations

from typing import Literal

from PIL import Image

from ..pool import ModelPool
from ..utils import validate_number_range
from .utils import binary_mask, blend_edit, generate_klein_edit

# Context Klein sees around the object in local mode (fraction of the bbox, >= 16px).
LOCAL_CONTEXT = 0.2


# Distilled Klein follows short, positive descriptions better than "don't" lists.
# Kept generic: the wording must work for any object and any kind of edit.
COMPLETE_AND_REALISTIC = (
    "Apply the change to every part of the object, including thin or small "
    "parts. When changing only color, keep the same surface: recolor its "
    "existing material, texture, shading, highlights and shadows, and keep the "
    "original brightness and lighting, instead of painting a flat or glowing "
    "color over it."
)


def _editing_prompt(
    prompt: str,
    mode: Literal["local", "regional", "global"],
    bounds: tuple[int, int, int, int],
) -> str:
    if mode == "local":
        instructions = (
            "Edit only the selected object's appearance. Preserve its identity, "
            "silhouette, geometry, size, position, and pose. Do not add or remove "
            "parts. Keep everything outside the selected object unchanged. "
            + COMPLETE_AND_REALISTIC
        )
    elif mode == "regional":
        instructions = (
            "Edit the selected object and allow the requested change to extend into "
            "the padded area around it, including changes to its silhouette, size, "
            "or additions such as accessories. Keep image content outside that "
            "nearby edit area unchanged. " + COMPLETE_AND_REALISTIC
        )
    else:
        instructions = (
            f"The selected object is approximately inside pixel box {list(bounds)}. "
            "Apply the requested edit using the full image for context. Change the "
            "selected object and any surrounding scene elements needed for the "
            "request; preserve unrelated content."
        )
    return f"{instructions}\nRequested edit: {prompt.strip()}"


def flux2_klein_edit_object(
    pool: ModelPool,
    image: Image.Image,
    mask: Image.Image,
    prompt: str,
    *,
    mode: Literal["local", "regional", "global"] = "local",
    padding: float = 0.2,
    debug: dict | None = None,
) -> Image.Image:
    """Edit one masked object with Klein and paste it back with a ViTMatte matte.

    - ``global``: the whole image goes to Klein and its result is returned.
    - ``local`` / ``regional``: Klein edits a crop around the object (extra
      context in local mode, ``padding`` of the bbox in regional mode). The
      matte of Klein's output, with a trimap seeded by the mask, is the paste
      weight; nothing else is applied.
    """
    if not isinstance(image, Image.Image):
        raise TypeError("image must be a Pillow image")
    if not isinstance(mask, Image.Image):
        raise TypeError("each mask must be a Pillow image")
    if mask.size != image.size:
        raise ValueError("each mask must have the same size as the image")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("each prompt must be a non-empty string")
    if mode not in ("local", "regional", "global"):
        raise ValueError("mode must be 'local', 'regional', or 'global'")
    validate_number_range("padding", padding, 0, 1)
    mask = mask.convert("L")
    bounds = mask.getbbox()
    if bounds is None:
        raise ValueError("mask is empty")
    left, top, right, bottom = bounds
    width, height = right - left, bottom - top

    current = image.convert("RGB")
    instruction = _editing_prompt(prompt, mode, bounds)
    pipeline = pool.flux2_klein_pipeline
    config = pool.config.klein

    if mode == "global":
        edited = generate_klein_edit(pipeline, current, instruction, config)
        if edited.size != current.size:
            edited = edited.resize(current.size, Image.Resampling.LANCZOS)
        return edited

    if mode == "regional":
        margin_x, margin_y = round(width * padding), round(height * padding)
    else:
        margin_x = max(16, round(width * LOCAL_CONTEXT))
        margin_y = max(16, round(height * LOCAL_CONTEXT))
    crop_box = (
        max(0, left - margin_x),
        max(0, top - margin_y),
        min(image.width, right + margin_x),
        min(image.height, bottom + margin_y),
    )

    before = current.crop(crop_box)
    after = generate_klein_edit(pipeline, before, instruction, config)
    if after.size != before.size:
        after = after.resize(before.size, Image.Resampling.LANCZOS)

    object_mask = binary_mask(mask.crop(crop_box))
    result, matte = blend_edit(pool, current, crop_box, after, object_mask, debug=debug)
    if debug is not None:
        debug.update(before=before, klein=after, matte=matte, minimum=object_mask)
    return result
