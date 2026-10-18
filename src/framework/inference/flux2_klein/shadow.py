"""Find the cast shadow of an object so it can be removed together with it."""

from __future__ import annotations

import cv2
import numpy as np
from PIL import Image

from ..pool import ModelPool
from ..sam3 import sam3_segment_masks

SHADOW_PROMPT = "shadow"
MIN_SHADOW_CONFIDENCE = 0.3
# A shadow bigger than the object is a scene shadow (e.g. a bench's), not its own.
MAX_AREA_RATIO = 1.0
# The object's own shadow lies within this distance of its bbox (fraction of size).
WINDOW_RATIO = 0.35
# A SAM3 segment with more than this share outside the window is a scene shadow.
MAX_OUTSIDE_SHARE = 0.25


def _dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return mask
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1)
    )
    return cv2.dilate(mask, kernel)


def _touching_components(candidate: np.ndarray, near: np.ndarray) -> np.ndarray:
    """Keep connected components of ``candidate`` that touch ``near``."""
    count, labels = cv2.connectedComponents(candidate.astype(np.uint8), connectivity=8)
    keep = np.unique(labels[(near > 0) & (candidate > 0)])
    keep = keep[keep != 0]
    return (
        np.isin(labels, keep) if count > 1 and len(keep) else np.zeros_like(near, bool)
    )


def _sam3_shadow(
    pool: ModelPool, crop: Image.Image, near: np.ndarray, window: np.ndarray
) -> np.ndarray:
    """Union of SAM3 "shadow" segments that touch the object and stay local to it."""
    try:
        masks = sam3_segment_masks(
            pool, crop, SHADOW_PROMPT, min_confidence=MIN_SHADOW_CONFIDENCE
        )
    except Exception:
        return np.zeros(near.shape, bool)
    union = np.zeros(near.shape, bool)
    for mask in masks:
        array = np.asarray(mask.convert("L")) > 127
        if array.shape != near.shape or not (array & (near > 0)).any():
            continue
        outside = (array & ~window).sum() / max(1, array.sum())
        if outside <= MAX_OUTSIDE_SHARE:
            union |= array & window
    return union


def _heuristic_shadow(
    crop: Image.Image, obj: np.ndarray, near: np.ndarray, size: int, window: np.ndarray
) -> np.ndarray:
    """Pixels clearly darker than their surroundings that touch the object."""
    lightness = cv2.cvtColor(np.asarray(crop), cv2.COLOR_RGB2LAB)[..., 0].astype(
        np.float32
    )
    ring = (_dilate(obj, round(size * 0.6)) > 0) & (near == 0)
    if ring.sum() < 64:
        return np.zeros(obj.shape, bool)
    reference = float(np.percentile(lightness[ring], 70))
    dark = ((lightness < reference - max(12.0, 0.2 * reference)) & (obj == 0)).astype(
        np.uint8
    )
    dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    return _touching_components(dark, near) & window


def find_shadow_mask(
    pool: ModelPool, crop: Image.Image, object_mask: np.ndarray
) -> np.ndarray:
    """Return a 0/255 mask of the object's shadow inside ``crop``.

    SAM3 segments "shadow" and keeps segments that touch the object and stay
    within a window around it (large scene shadows are rejected). When
    SAM3 finds none, a luminance heuristic picks dark pixels next to the object.
    The result excludes the object itself and may be empty.
    """
    obj = (object_mask > 0).astype(np.uint8)
    ys, xs = np.nonzero(obj)
    if len(ys) == 0:
        return np.zeros(obj.shape, np.uint8)
    size = max(xs.max() - xs.min() + 1, ys.max() - ys.min() + 1)
    touch = max(3, round(size * 0.03))
    near = _dilate(obj, touch)
    window = np.zeros(obj.shape, bool)
    window[
        max(0, ys.min() - round(size * WINDOW_RATIO)) : ys.max()
        + round(size * WINDOW_RATIO)
        + 1,
        max(0, xs.min() - round(size * WINDOW_RATIO)) : xs.max()
        + round(size * WINDOW_RATIO)
        + 1,
    ] = True

    shadow = _sam3_shadow(pool, crop, near, window)
    if not shadow.any():
        shadow = _heuristic_shadow(crop, obj, near, int(size), window)
    shadow &= obj == 0
    if shadow.sum() > MAX_AREA_RATIO * float(obj.sum()):
        return np.zeros(obj.shape, np.uint8)
    return _dilate(shadow.astype(np.uint8), touch) * 255
