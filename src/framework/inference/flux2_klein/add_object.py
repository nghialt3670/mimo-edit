"""Production editing and insertion helpers for FLUX.2 Klein 4B."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Literal, Sequence

import numpy as np
import torch
from PIL import Image, ImageChops, ImageFilter, ImageOps
from scipy.ndimage import maximum_filter

from framework.utils.randomness import get_random_seed

from framework.inference.config import KleinInferenceConfig
from framework.inference.pool import ModelPool

if TYPE_CHECKING:
    from diffusers import Flux2KleinPipeline


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
    pool: ModelPool,
    image: Image.Image,
    box: Sequence[float],
    prompt: str,
    **kwargs,
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
        inference_config = pool.config.klein or KleinInferenceConfig()
        edited = (
            pool.flux2_klein_pipeline(
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
    pool: ModelPool,
    image: Image.Image,
    mask: Image.Image,
    prompt: str,
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

    inference_config = pool.config.klein or KleinInferenceConfig()
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
        pool.flux2_klein_pipeline,
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
