"""Carry Klein's cast shadow of a newly added object over to the original scene."""

from __future__ import annotations

import numpy as np
from PIL import Image
from scipy.ndimage import (
    binary_closing,
    binary_fill_holes,
    distance_transform_edt,
    gaussian_filter,
    label,
    shift,
)

MAX_DARKENING = 0.85
# Shadows are darker but keep their hue; a hue change means the content changed.
MAX_CHROMA_SHIFT = 0.12
DARKER_THAN = 0.9
# A cast shadow lies within this distance of the object (fraction of its size).
WINDOW_RATIO = 0.6
FALLBACK_STRENGTH = 0.35
MIN_SHADOW_SHARE = 0.02


def _luma(rgb: np.ndarray) -> np.ndarray:
    return rgb @ np.array([0.299, 0.587, 0.114], np.float32)


def transfer_shadow(
    before: Image.Image,
    after: Image.Image,
    matte: np.ndarray,
    *,
    ground: bool = False,
) -> tuple[np.ndarray, str]:
    """Multiplicative shade map (crop-sized, 1 = no change) for the original scene.

    The shade is Klein's luminance ratio ``after / before`` outside the object,
    cleaned of Klein's global exposure drift, restricted to darkening that keeps
    the colour and touches the object. The original texture is kept and only
    its lighting is taken. With ``ground`` and no shadow found, a soft contact
    shadow below the object is synthesised. Returns ``(shade, source)`` where
    source is ``"transfer"``, ``"synthetic"`` or ``"none"``.
    """
    none = np.ones(matte.shape, np.float32)
    obj = matte > 0.5
    if not obj.any():
        return none, "none"
    b = np.asarray(before.convert("RGB"), np.float32) + 1.0
    a = np.asarray(after.convert("RGB"), np.float32) + 1.0
    lb, la = _luma(b), _luma(a)
    ys, xs = np.nonzero(obj)
    size = float(max(np.ptp(ys), np.ptp(xs)) + 1)
    dist = distance_transform_edt(~obj)

    reference = dist > WINDOW_RATIO * size
    pool = reference if reference.sum() > 100 else ~obj
    ratio = gaussian_filter(la / lb / np.median((la / lb)[pool]), 1.5)
    chroma = gaussian_filter(np.abs(a / la[..., None] - b / lb[..., None]).max(-1), 1.5)
    near = max(3.0, 0.03 * size)
    candidate = (
        (ratio < DARKER_THAN)
        & (chroma < MAX_CHROMA_SHIFT)
        & (dist <= WINDOW_RATIO * size)
        & (matte < 0.05)
    )
    labels, count = label(candidate)
    touching = np.unique(labels[candidate & (dist <= near)])
    kept = np.isin(labels, touching[touching != 0]) if count else candidate & False
    kept = binary_fill_holes(binary_closing(kept, iterations=2)) & (matte < 0.05)

    if kept.sum() >= MIN_SHADOW_SHARE * obj.sum():
        weight = np.clip(
            gaussian_filter(kept.astype(np.float32), max(2.0, 0.02 * size)), 0, 1
        ) * (1.0 - matte)
        darkening = 1.0 - np.clip(ratio, 1.0 - MAX_DARKENING, 1.0)
        return (1.0 - weight * darkening).astype(np.float32), "transfer"
    if not ground:
        return none, "none"

    height = ys.max() - ys.min() + 1
    foot = obj & (np.arange(obj.shape[0])[:, None] > ys.max() - 0.3 * height)
    soft = gaussian_filter(
        shift(foot.astype(np.float32), (0.05 * height, 0), order=1), 0.05 * size
    )
    soft = np.clip(soft / max(float(soft.max()), 1e-6), 0, 1) * (matte < 0.05)
    return (1.0 - FALLBACK_STRENGTH * soft).astype(np.float32), "synthetic"
