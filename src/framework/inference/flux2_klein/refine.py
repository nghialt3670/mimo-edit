"""Klein pass over the whole composited image to blend a pasted object in."""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image
from scipy.ndimage import distance_transform_edt

from framework.utils.randomness import get_random_seed

from ..pool import ModelPool
from .utils import WORK_SIZE

REFINE_PROMPT = (
    "An object was pasted into this photo. Blend it into the scene: fix edge "
    "artifacts, match the lighting, colour, sharpness and perspective, and add "
    "natural shadows and reflections where it touches surfaces. Keep the object's "
    "identity and keep everything else exactly the same."
)
# Klein's output is used within this distance of the object (fraction of its size),
# fading out over FEATHER; farther away the composite is kept untouched.
KEEP_RADIUS = 0.5
FEATHER = 0.25


def refine_zone(matte: np.ndarray) -> np.ndarray:
    """Soft weight in [0, 1]: 1 on and near the object, 0 far from it."""
    obj = matte > 0.05
    if not obj.any():
        return np.zeros(matte.shape, np.float32)
    ys, xs = np.nonzero(obj)
    size = float(max(np.ptp(ys), np.ptp(xs)) + 1)
    dist = distance_transform_edt(~obj)
    return np.clip(1.0 - (dist - KEEP_RADIUS * size) / (FEATHER * size), 0, 1).astype(
        np.float32
    )


def klein_refine(
    pool: ModelPool, composite: Image.Image, matte: np.ndarray
) -> tuple[Image.Image, Image.Image]:
    """Run Klein on the whole ``composite``; return ``(result, raw_output)``.

    ``matte`` is full-image sized. The result takes Klein's pixels near the
    object and the untouched composite elsewhere, so the background cannot drift.
    """
    width, height = composite.size
    scale = WORK_SIZE / max(width, height)
    ww = max(64, round(width * scale / 16) * 16)
    wh = max(64, round(height * scale / 16) * 16)
    config = pool.config.klein
    with torch.inference_mode():
        raw = (
            pool.flux2_klein_pipeline(
                image=composite.resize((ww, wh), Image.Resampling.LANCZOS),
                prompt=REFINE_PROMPT,
                height=wh,
                width=ww,
                num_inference_steps=config.num_inference_steps,
                guidance_scale=config.guidance_scale,
                generator=torch.Generator(device="cpu").manual_seed(get_random_seed()),
            )
            .images[0]
            .convert("RGB")
        )
    raw = raw.resize((width, height), Image.Resampling.LANCZOS)
    zone = refine_zone(matte)[..., None]
    mixed = zone * np.asarray(raw, np.float32) + (1 - zone) * np.asarray(
        composite, np.float32
    )
    return Image.fromarray(mixed.round().astype(np.uint8), "RGB"), raw
