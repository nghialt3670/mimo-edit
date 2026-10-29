"""Edit an existing object with FLUX.2 Klein 4B and a consistency LoRA."""

from __future__ import annotations

from typing import Literal

from PIL import Image

from ..config import KleinInferenceConfig
from ..pool import ModelPool
from ..utils import validate_number_range
from .utils import binary_mask, blend_edit, generate_klein_edit

# Context Klein sees around the object in local mode (fraction of the bbox, >= 16px).
LOCAL_CONTEXT = 0.3


# Distilled Klein follows short, positive descriptions better than "don't" lists.
# Kept generic: the wording must work for any object and any kind of edit.
COMPLETE_AND_REALISTIC = (
    "Apply the change to every part of the object, including thin or small "
    "parts. When changing only color, keep the same surface: recolor its "
    "existing material, texture, shading, highlights and shadows, and keep the "
    "original brightness and lighting, instead of painting a flat or glowing "
    "color over it."
)


REFERENCE_GRAY = (128, 128, 128)
REFERENCE_LONG_SIDE = 512
REFERENCE_CLAUSE = (
    "The second image is a reference. Transfer only the attribute named in the "
    "requested edit from it onto the selected object in the first image. Keep the "
    "selected object's own shape, identity, position and texture unless the request "
    "asks otherwise; do not copy the reference's shape or content."
)


def _reference_tile(reference: Image.Image, mask: Image.Image | None) -> Image.Image:
    """Masked reference on neutral gray, cropped to the mask and downscaled for Klein."""
    rgb = reference.convert("RGB")
    if mask is None:
        tile = rgb
    else:
        selection = mask.convert("L")
        if selection.getextrema()[1] <= 1:
            selection = selection.point(lambda value: 255 if value else 0)
        bounds = selection.getbbox()
        if bounds is None:
            raise ValueError("reference_mask is empty")
        background = Image.new("RGB", rgb.size, REFERENCE_GRAY)
        tile = Image.composite(rgb, background, selection).crop(bounds)
    scale = min(1.0, REFERENCE_LONG_SIDE / max(tile.size))
    if scale < 1.0:
        tile = tile.resize(
            (max(1, round(tile.width * scale)), max(1, round(tile.height * scale))),
            Image.Resampling.LANCZOS,
        )
    return tile


def _generate_edit(
    pipeline,
    config: KleinInferenceConfig,
    source: Image.Image,
    instruction: str,
    lora_strength: float,
    reference: Image.Image | None,
) -> Image.Image:
    inputs = source if reference is None else [source, reference]
    return generate_klein_edit(
        pipeline, inputs, instruction, config, lora_strength=lora_strength
    )


def _editing_prompt(
    prompt: str,
    mode: Literal["local", "regional", "global"],
    bounds: tuple[int, int, int, int],
    *,
    reference: bool = False,
) -> str:
    if mode == "local":
        instructions = (
            "Edit only the selected object's appearance. Preserve its identity, "
            "silhouette, geometry, size, position, and pose. Do not add or remove "
            "parts. Keep everything outside the selected object unchanged. "
            
        )
    elif mode == "regional":
        instructions = (
            "Edit the selected object and allow the requested change to extend into "
            "the padded area around it, including changes to its silhouette, size, "
            "or additions such as accessories. Keep image content outside that "
            "nearby edit area unchanged. " 
        )
    else:
        instructions = (
            f"The selected object is approximately inside pixel box {list(bounds)}. "
            "Apply the requested edit using the full image for context. Change the "
            "selected object and any surrounding scene elements needed for the "
            "request; preserve unrelated content."
        )
    if reference:
        instructions = instructions.rstrip() + " " + REFERENCE_CLAUSE + " "
    return f"{instructions}\n{COMPLETE_AND_REALISTIC}\nRequested edit: {prompt.strip()}"


def flux2_klein_edit_object(
    pool: ModelPool,
    image: Image.Image,
    mask: Image.Image,
    prompt: str,
    *,
    mode: Literal["local", "regional", "global"] = "local",
    padding: float = 0.2,
    lora_strength: float = 0.6,
    reference_image: Image.Image | None = None,
    reference_mask: Image.Image | None = None,
    debug: dict | None = None,
) -> Image.Image:
    """Edit the masked object using the FLUX.2 Klein 4B consistency LoRA.

    The consistency LoRA is loaded once onto the shared Klein pipeline and
    only switched on for this call. Removal keeps its own named adapter;
    neither call unloads the other. Set ``lora_strength=0`` to run the base
    model for A/B comparisons.

    ``local`` crops around the object and restricts pasted changes to the mask.
    ``regional`` allows surrounding changes through semantic-difference blending.
    ``global`` returns the full Klein-generated image.
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
    validate_number_range("lora_strength", lora_strength, 0, 1)

    mask = mask.convert("L")
    bounds = mask.getbbox()
    if bounds is None:
        raise ValueError("mask is empty")
    left, top, right, bottom = bounds
    width, height = right - left, bottom - top

    reference_tile = None
    if reference_image is not None or reference_mask is not None:
        reference_source = image if reference_image is None else reference_image
        if not isinstance(reference_source, Image.Image):
            raise TypeError("reference_image must be a Pillow image")
        if reference_mask is not None:
            if not isinstance(reference_mask, Image.Image):
                raise TypeError("reference_mask must be a Pillow image")
            if reference_mask.size != reference_source.size:
                raise ValueError("reference_mask must match the reference image size")
        reference_tile = _reference_tile(reference_source, reference_mask)

    current = image.convert("RGB")
    instruction = _editing_prompt(
        prompt, mode, bounds, reference=reference_tile is not None
    )
    pipeline = pool.flux2_klein_pipeline
    config = pool.config.klein

    if mode == "global":
        edited = _generate_edit(
            pipeline, config, current, instruction, lora_strength, reference_tile
        )
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
    after = _generate_edit(
        pipeline, config, before, instruction, lora_strength, reference_tile
    )
    if after.size != before.size:
        after = after.resize(before.size, Image.Resampling.LANCZOS)

    object_mask = binary_mask(mask.crop(crop_box))
    result, matte = blend_edit(pool, current, crop_box, after, object_mask, debug=debug)
    if debug is not None:
        debug.update(before=before, klein=after, matte=matte, minimum=object_mask)
    return result
