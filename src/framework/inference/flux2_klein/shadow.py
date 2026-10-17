"""Find the cast shadow of an object so it can be removed together with it."""

from __future__ import annotations

import cv2
import numpy as np
from PIL import Image

from ..pool import ModelPool
from ..sam3 import sam3_segment_masks

SHADOW_PROMPT = "shadow"
MIN_SHADOW_CONFIDENCE = 0.3
# A "shadow" much larger than the object is almost surely a misdetection
# (e.g. a dark floor), so it is ignored instead of erased.
MAX_AREA_RATIO = 3.0


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
    pool: ModelPool, crop: Image.Image, near: np.ndarray, max_area: float
) -> np.ndarray:
    try:
        masks = sam3_segment_masks(
            pool, crop, SHADOW_PROMPT, min_confidence=MIN_SHADOW_CONFIDENCE
        )
    except Exception:
        return np.zeros(near.shape, bool)
    union = np.zeros(near.shape, bool)
    for mask in masks:
        array = np.asarray(mask.convert("L")) > 127
        if array.shape == near.shape and (array & (near > 0)).any():
            union |= array
    return union if union.sum() <= max_area else np.zeros(near.shape, bool)


def _heuristic_shadow(
    crop: Image.Image, obj: np.ndarray, near: np.ndarray, size: int, max_area: float
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
    shadow = _touching_components(dark, near)
    return shadow if shadow.sum() <= max_area else np.zeros(obj.shape, bool)


def find_shadow_mask(
    pool: ModelPool, crop: Image.Image, object_mask: np.ndarray
) -> np.ndarray:
    """Return a 0/255 mask of the object's shadow inside ``crop``.

    SAM3 segments "shadow" and keeps the segments touching the object. When
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
    max_area = MAX_AREA_RATIO * float(obj.sum())

    shadow = _sam3_shadow(pool, crop, near, max_area)
    if not shadow.any():
        shadow = _heuristic_shadow(crop, obj, near, int(size), max_area)
    shadow &= obj == 0
    return _dilate(shadow.astype(np.uint8), touch) * 255
