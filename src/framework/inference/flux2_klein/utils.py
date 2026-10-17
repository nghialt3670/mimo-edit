"""Helpers shared by the FLUX.2 Klein editing, insertion, and removal functions."""

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

import numpy as np
import torch
from PIL import Image

from framework.utils.randomness import get_random_seed

from ..config import KleinInferenceConfig

if TYPE_CHECKING:
    from diffusers import Flux2KleinPipeline


def harmonize_with_mask(predictor, image: Image.Image, mask: Image.Image):
    from ..external.pctnet_harmonization import harmonize_pctnet

    return harmonize_pctnet(predictor, image, mask)


def harmonize_changed_region(
    predictor,
    before: Image.Image,
    after: Image.Image,
    box: Sequence[float],
) -> Image.Image:
    """Harmonize pixels that changed between before/after inside an XYXY box."""
    from ..external.pctnet_harmonization import harmonize_pctnet_with_box

    try:
        return harmonize_pctnet_with_box(predictor, before, after, box)
    except ValueError:
        return after


def generate_klein_edit(
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
PADDING_FRAC = 0
WORK_SIZE = 1024
EDGE_BLEND_PX = 12


def edge_padded_crop(image: Image.Image, box: tuple[int, int, int, int]) -> Image.Image:
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


def feather_mask(size: tuple[int, int]) -> Image.Image:
    """Smoothstep fade at the edges of an edited rectangular crop."""
    width, height = size
    if EDGE_BLEND_PX <= 0:
        return Image.new("L", size, 255)
    x = np.minimum(np.arange(width), np.arange(width)[::-1])[None, :]
    y = np.minimum(np.arange(height), np.arange(height)[::-1])[:, None]
    alpha = np.clip(np.minimum(x, y) / EDGE_BLEND_PX, 0, 1)
    alpha = alpha * alpha * (3 - 2 * alpha)
    return Image.fromarray(np.uint8(np.rint(alpha * 255)), "L")
