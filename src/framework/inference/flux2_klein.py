"""Production editing and insertion helpers for FLUX.2 Klein 4B."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Literal, Sequence

import numpy as np
import torch
from PIL import Image, ImageChops, ImageFilter, ImageOps
from scipy.ndimage import maximum_filter

from framework.utils.randomness import get_random_seed

from .config import KleinInferenceConfig
from .pool import ModelPool
from .utils import validate_number_range

if TYPE_CHECKING:
    from diffusers import Flux2KleinPipeline


def _editing_prompt(
    prompt: str,
    mode: Literal["local", "regional", "global"],
    bounds: tuple[int, int, int, int],
) -> str:
    if mode == "local":
        instructions = (
            "Edit only the selected object's appearance. Preserve its identity, "
            "silhouette, geometry, size, position, and pose. Do not add or remove "
            "parts. Keep everything outside the selected object unchanged."
        )
    elif mode == "regional":
        instructions = (
            "Edit the selected object and allow the requested change to extend into "
            "the padded area around it, including changes to its silhouette, size, "
            "or additions such as accessories. Keep image content outside that "
            "nearby edit area unchanged."
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
    padding: float = 0.25,
) -> Image.Image:
    """Edit one masked object and feather the local/regional composite boundary."""
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
    current = image.convert("RGB")
    instruction = _editing_prompt(prompt, mode, bounds)

    pipeline = pool.flux2_klein_pipeline
    config = pool.config.klein
    if mode == "global":
        edited = _generate_klein_edit(
            pipeline, current, instruction, config
        )
        if edited.size != current.size:
            edited = edited.resize(current.size, Image.Resampling.LANCZOS)
        return edited.convert("RGB")

    if mode == "regional":
        padding_x = round((right - left) * padding)
        padding_y = round((bottom - top) * padding)
        allowed_mask = Image.fromarray(
            maximum_filter(
                np.asarray(mask),
                size=(padding_y * 2 + 1, padding_x * 2 + 1),
                mode="constant",
                cval=0,
            )
        )
    else:
        padding_x = max(16, round((right - left) * 0.2))
        padding_y = max(16, round((bottom - top) * 0.2))
        allowed_mask = mask

    feather_radius = min(12.0, max(2.0, min(right - left, bottom - top) * 0.04))
    feather_support = int(np.ceil(feather_radius * 3))
    allowed_bounds = allowed_mask.getbbox()
    if allowed_bounds is None:
        raise ValueError("mask is empty")
    allowed_left, allowed_top, allowed_right, allowed_bottom = allowed_bounds
    crop_padding_x = feather_support + (padding_x if mode == "local" else 0)
    crop_padding_y = feather_support + (padding_y if mode == "local" else 0)
    crop_box = (
        max(0, allowed_left - crop_padding_x),
        max(0, allowed_top - crop_padding_y),
        min(image.width, allowed_right + crop_padding_x),
        min(image.height, allowed_bottom + crop_padding_y),
    )
    original_crop = current.crop(crop_box)
    feathered_mask = allowed_mask.filter(
        ImageFilter.GaussianBlur(radius=feather_radius)
    )
    composite_mask = ImageChops.lighter(allowed_mask, feathered_mask)
    edited_crop = _generate_klein_edit(pipeline, original_crop, instruction, config)
    if edited_crop.size != original_crop.size:
        edited_crop = edited_crop.resize(original_crop.size, Image.Resampling.LANCZOS)

    current.paste(edited_crop, crop_box[:2], composite_mask.crop(crop_box))
    return current


def _generate_klein_edit(
    pipeline: Flux2KleinPipeline,
    image: Image.Image,
    prompt: str,
    config: KleinInferenceConfig,
) -> Image.Image:
    with torch.inference_mode():
        return (
            pipeline(
                image=image,
                prompt=prompt,
                num_inference_steps=config.num_inference_steps,
                guidance_scale=config.guidance_scale,
                generator=torch.Generator(device="cpu").manual_seed(get_random_seed()),
            )
            .images[0]
            .convert("RGB")
        )


# Shared settings for distilled FLUX.2 Klein 4B.
PADDING_FRAC = 0.60
WORK_SIZE = 1024
EDGE_BLEND_PX = 12


def _edge_padded_crop(
    image: Image.Image, box: tuple[int, int, int, int]
) -> Image.Image:
    """Crop a possibly out-of-bounds XYXY box, repeating boundary pixels."""
    x1, y1, x2, y2 = box
    w, h = image.size
    valid = (max(0, x1), max(0, y1), min(w, x2), min(h, y2))
    vx1, vy1, vx2, vy2 = valid
    return Image.fromarray(
        np.pad(
            np.asarray(image.crop(valid)),
            ((vy1 - y1, y2 - vy2), (vx1 - x1, x2 - vx2), (0, 0)),
            mode="edge",
        )
    )


def _feather_mask(size: tuple[int, int]) -> Image.Image:
    """Smoothstep fade at the edges of an edited rectangular crop."""
    width, height = size
    if EDGE_BLEND_PX <= 0:
        return Image.new("L", size, 255)
    x = np.minimum(np.arange(width), np.arange(width)[::-1])[None, :]
    y = np.minimum(np.arange(height), np.arange(height)[::-1])[:, None]
    alpha = np.clip(np.minimum(x, y) / EDGE_BLEND_PX, 0, 1)
    alpha = alpha * alpha * (3 - 2 * alpha)
    return Image.fromarray(np.uint8(np.rint(alpha * 255)), "L")


def flux2_klein_add_object_with_box(
    pipe: Flux2KleinPipeline,
    image: Image.Image,
    box: Sequence[float],
    prompt: str,
    *,
    config: KleinInferenceConfig | None = None,
) -> Image.Image:
    """Add an object near an XYXY pixel box and paste the entire edited crop."""
    if not isinstance(image, Image.Image):
        raise TypeError("image must be a PIL image")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("prompt must be a nonempty string")
    if len(box) != 4 or not all(math.isfinite(float(v)) for v in box):
        raise ValueError("box must contain four finite XYXY coordinates")

    original = image.convert("RGB")
    width, height = original.size
    x1, y1, x2, y2 = map(
        int,
        np.clip(
            [round(float(v)) for v in box],
            [0, 0, 0, 0],
            [width, height, width, height],
        ),
    )
    if x1 >= x2 or y1 >= y2:
        raise ValueError("box must have positive area inside the image")

    pad = math.ceil(max(x2 - x1, y2 - y1) * PADDING_FRAC)
    crop_box = (x1 - pad, y1 - pad, x2 + pad, y2 + pad)
    cropped = _edge_padded_crop(original, crop_box)
    cw, ch = cropped.size
    scale = WORK_SIZE / max(cw, ch)
    ww = max(64, round(cw * scale / 16) * 16)
    wh = max(64, round(ch * scale / 16) * 16)

    with torch.inference_mode():
        inference_config = config or KleinInferenceConfig()
        edited = (
            pipe(
                image=cropped.resize((ww, wh), Image.Resampling.LANCZOS),
                prompt=f"Add {prompt} to the center.",
                height=wh,
                width=ww,
                num_inference_steps=inference_config.num_inference_steps,
                guidance_scale=inference_config.guidance_scale,
                generator=torch.Generator(device="cpu").manual_seed(get_random_seed()),
            )
            .images[0]
            .convert("RGB")
        )

    if edited.size != (ww, wh):
        raise ValueError(f"Klein returned {edited.size}, expected {(ww, wh)}")

    blended = Image.composite(
        edited.resize((cw, ch), Image.Resampling.LANCZOS),
        cropped,
        _feather_mask((cw, ch)),
    )
    cx1, cy1, cx2, cy2 = crop_box
    vx1, vy1 = max(cx1, 0), max(cy1, 0)
    vx2, vy2 = min(cx2, width), min(cy2, height)
    result = original.copy()
    result.paste(
        blended.crop((vx1 - cx1, vy1 - cy1, vx2 - cx1, vy2 - cy1)),
        (vx1, vy1),
    )
    return result


def flux2_klein_add_object_with_mask(
    pipe: Flux2KleinPipeline,
    image: Image.Image,
    mask: Image.Image,
    prompt: str,
    *,
    config: KleinInferenceConfig | None = None,
) -> Image.Image:
    """Generate inside a white mask, preserving every pixel outside the mask."""
    if not isinstance(image, Image.Image) or not isinstance(mask, Image.Image):
        raise TypeError("image and mask must both be PIL images")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("prompt must be a nonempty string")
    if mask.size != image.size:
        raise ValueError("mask must have the same size as image")

    original = image.convert("RGB")
    mask = mask.convert("L")
    if mask.getextrema()[1] <= 1:
        mask = mask.point(lambda value: value * 255)
    mask = mask.point(lambda value: 255 if value >= 128 else 0)
    region = mask.getbbox()
    if region is None:
        raise ValueError("mask is empty")

    inference_config = config or KleinInferenceConfig()
    left, top, right, bottom = region
    padding = inference_config.mask_context_padding
    crop_box = (
        max(0, left - padding),
        max(0, top - padding),
        min(original.width, right + padding),
        min(original.height, bottom + padding),
    )
    original_crop = original.crop(crop_box)
    working_image = ImageOps.contain(original_crop, (WORK_SIZE, WORK_SIZE))

    edited = _generate_klein_edit(
        pipe,
        working_image,
        f"Add {prompt} inside the selected region. Preserve the surroundings.",
        inference_config,
    )
    if edited.size != working_image.size:
        raise ValueError(f"Klein returned {edited.size}, expected {working_image.size}")

    crop_mask = mask.crop(crop_box)
    edited_crop = edited.resize(original_crop.size, Image.Resampling.LANCZOS)
    result = original.copy()
    result.paste(edited_crop, crop_box[:2], crop_mask)
    return result


__all__ = [
    "flux2_klein_add_object_with_box",
    "flux2_klein_add_object_with_mask",
    "flux2_klein_edit_object",
]
