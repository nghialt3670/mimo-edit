"""Inference bridge: instantiate a SAM3 segmenter using the SAME ModelPool."""

from __future__ import annotations

import cv2
import numpy as np
from PIL import Image

from framework.utils.randomness import get_random_seed

from .pool import ModelPool


def object_clear_remove_object(
    pool: ModelPool,
    image: Image.Image,
    mask: Image.Image,
    prompt: str = "remove the selected object",
) -> Image.Image:
    import torch

    config = pool.config.object_clear
    source = image.convert("RGB")
    normalized_mask = mask.convert("L").resize(source.size, Image.Resampling.NEAREST)
    normalized_mask = _pad_mask(normalized_mask, config.mask_padding_ratio)
    original_size = source.size
    scale = config.working_min_side / min(source.size)
    working_size = (
        max(1, round(source.width * scale)),
        max(1, round(source.height * scale)),
    )
    pipeline = pool.object_clear_pipeline
    result = pipeline(
        prompt=prompt or "remove the instance of object",
        image=source.resize(working_size, Image.Resampling.BILINEAR),
        mask_image=normalized_mask.resize(working_size, Image.Resampling.NEAREST),
        generator=torch.Generator(device=pipeline.unet.device).manual_seed(
            get_random_seed()
        ),
        num_inference_steps=config.num_inference_steps,
        guidance_scale=config.guidance_scale,
        height=working_size[1],
        width=working_size[0],
        return_attn_map=False,
    )
    return (
        result.images[0].resize(original_size, Image.Resampling.BICUBIC).convert("RGB")
    )


def _pad_mask(mask: Image.Image, ratio: float) -> Image.Image:
    """Dilate the mask by `ratio` of its bounding-box longer side."""
    binary = mask.point(lambda value: 255 if value >= 128 else 0)
    bbox = binary.getbbox()
    if bbox is None or ratio <= 0:
        return binary
    radius = round(max(bbox[2] - bbox[0], bbox[3] - bbox[1]) * ratio)
    if radius < 1:
        return binary
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1)
    )
    return Image.fromarray(cv2.dilate(np.asarray(binary), kernel), mode="L")


__all__ = [
    "object_clear_remove_object",
]
