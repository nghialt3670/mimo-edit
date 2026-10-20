"""Trimap-guided alpha matting with ViTMatte."""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image
from scipy.ndimage import distance_transform_edt

from .pool import ModelPool

FG, UNKNOWN, BG = 255, 128, 0


def make_trimap(mask: np.ndarray, band: float) -> np.ndarray:
    """Trimap from an object mask: eroded core, uncertain band, known background."""
    inside = distance_transform_edt(mask)
    outside = distance_transform_edt(~mask)
    trimap = np.full(mask.shape, BG, np.uint8)
    trimap[outside <= band] = UNKNOWN
    trimap[inside > band] = FG
    return trimap


def default_band(mask: np.ndarray) -> float:
    ys, xs = np.nonzero(mask)
    size = max(np.ptp(ys), np.ptp(xs)) + 1 if len(ys) else 1
    return float(np.clip(0.06 * size, 8.0, 48.0))


def vitmatte_alpha(
    pool: ModelPool, image: Image.Image, mask: np.ndarray, band: float | None = None
) -> np.ndarray:
    """Float alpha matte in [0, 1] of the object ``mask`` selects in ``image``.

    The mask only constrains the matte (core is foreground, far outside is
    background); fur, hair and soft edges are resolved inside the uncertain band.
    """
    if not mask.any():
        return np.zeros(mask.shape, np.float32)
    processor, model = pool.vitmatte_model
    width, height = image.size
    scale = min(1.0, pool.config.vitmatte.max_side / max(width, height))
    size = (max(32, round(width * scale)), max(32, round(height * scale)))
    trimap = make_trimap(mask, (band or default_band(mask)) * scale)
    if size != (width, height):
        trimap = np.asarray(
            Image.fromarray(trimap).resize(size, Image.Resampling.NEAREST)
        )
    inputs = processor(
        images=image.convert("RGB").resize(size, Image.Resampling.LANCZOS),
        trimaps=Image.fromarray(trimap),
        return_tensors="pt",
    )
    inputs = {key: value.to(model.device) for key, value in inputs.items()}
    inputs["pixel_values"] = inputs["pixel_values"].to(model.dtype)
    with torch.inference_mode():
        alphas = model(**inputs).alphas
    alpha = alphas[0, 0, : size[1], : size[0]].float().clamp(0, 1).cpu().numpy()
    if size != (width, height):
        alpha = np.asarray(
            Image.fromarray(alpha).resize((width, height), Image.Resampling.BICUBIC)
        ).clip(0, 1)
    # Known regions are exact: trust the trimap, not the network.
    full = make_trimap(mask, band or default_band(mask))
    alpha = np.where(full == FG, 1.0, np.where(full == BG, 0.0, alpha))
    return alpha.astype(np.float32)
