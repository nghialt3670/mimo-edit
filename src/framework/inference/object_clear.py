"""Inference bridge: instantiate a SAM3 segmenter using the SAME ModelPool."""

from __future__ import annotations

from PIL import Image

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
    original_size = source.size
    scale = config.working_min_side / min(source.size)
    working_size = (
        max(1, round(source.width * scale)),
        max(1, round(source.height * scale)),
    )
    pipeline = pool.object_clear
    result = pipeline(
        prompt=prompt or "remove the instance of object",
        image=source.resize(working_size, Image.Resampling.BILINEAR),
        mask_image=normalized_mask.resize(working_size, Image.Resampling.NEAREST),
        generator=torch.Generator(device=pipeline.unet.device).manual_seed(config.seed),
        num_inference_steps=config.num_inference_steps,
        guidance_scale=config.guidance_scale,
        height=working_size[1],
        width=working_size[0],
        return_attn_map=False,
    )
    return (
        result.images[0].resize(original_size, Image.Resampling.BICUBIC).convert("RGB")
    )


__all__ = [
    "object_clear_remove_object",
]
