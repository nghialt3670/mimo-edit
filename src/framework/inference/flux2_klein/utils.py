"""Helpers shared by the FLUX.2 Klein editing, insertion, and removal functions."""

from __future__ import annotations

from typing import TYPE_CHECKING

import cv2
import numpy as np
import torch
from PIL import Image
from scipy.ndimage import (
    binary_fill_holes,
    distance_transform_edt,
    label,
    maximum_filter,
)
from skimage.filters import threshold_otsu

from framework.utils.randomness import get_random_seed

from ..config import KleinInferenceConfig

if TYPE_CHECKING:
    from diffusers import Flux2KleinPipeline


def harmonize_with_mask(predictor, image: Image.Image, mask: Image.Image):
    from ..external.pctnet_harmonization import harmonize_pctnet

    return harmonize_pctnet(predictor, image, mask)


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

# Smallest LAB color distance that counts as a real edit, whatever the image.
MIN_CHANGE = 8.0
# Pixels fully changed at the Otsu threshold; ramp starts at this fraction of it.
RAMP_START = 0.4
# Without a mask, changed blobs smaller than this are noise.
MIN_BLOB_PIXELS = 25


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


def binary_mask(mask: Image.Image) -> np.ndarray:
    """Boolean array of a 0/255 or 0/1 mask."""
    array = np.asarray(mask.convert("L"))
    return array >= (1 if int(array.max()) <= 1 else 128)


def grow(region: np.ndarray, rx: int, ry: int) -> np.ndarray:
    """Rectangular dilation by ``rx`` columns and ``ry`` rows."""
    return maximum_filter(region, size=(2 * ry + 1, 2 * rx + 1), mode="constant")


def feather_radius(width: int, height: int) -> float:
    """Feather size (px) for an object of the given bounding-box size."""
    return min(12.0, max(2.0, min(width, height) * 0.04))


def background_drift(
    delta: np.ndarray, reference: np.ndarray, sigma: float
) -> np.ndarray:
    """Smooth field of Klein's global tone shift, from unedited pixels only."""
    weight = np.maximum(
        cv2.GaussianBlur(reference.astype(np.float32), (0, 0), sigma), 1e-3
    )
    return (
        np.stack(
            [
                cv2.GaussianBlur(delta[..., c] * reference, (0, 0), sigma)
                for c in range(3)
            ],
            axis=-1,
        )
        / weight[..., None]
    )


def change_residual(
    before: Image.Image,
    after: Image.Image,
    minimum: np.ndarray,
    limit: np.ndarray,
    reach: float | None,
) -> tuple[np.ndarray, float]:
    """Per-pixel size of Klein's real edit after removing its global tone drift.

    Returns the residual and the magnitude of the estimated drift.
    """
    lab = [
        cv2.cvtColor(np.asarray(img.convert("RGB")), cv2.COLOR_RGB2LAB).astype(
            np.float32
        )
        for img in (before, after)
    ]
    delta = lab[1] - lab[0]
    magnitude = np.linalg.norm(delta, axis=-1)
    values = magnitude[limit] if limit.any() else magnitude.ravel()
    has_minimum = bool(minimum.any())
    distance = distance_transform_edt(~minimum) if has_minimum else None

    unedited = magnitude < (
        float(threshold_otsu(values)) if np.ptp(values) > 1e-3 else MIN_CHANGE
    )
    if has_minimum:
        unedited &= distance > (reach or 0)
    sigma = max(8.0, 0.1 * max(magnitude.shape))
    drift = background_drift(delta, unedited, sigma)
    residual = cv2.GaussianBlur(np.linalg.norm(delta - drift, axis=-1), (0, 0), 0.7)

    drift_size = (
        float(np.linalg.norm(drift[unedited].mean(axis=0))) if unedited.any() else 0.0
    )
    return residual, drift_size


def matte_alpha(
    matte: np.ndarray,
    residual: np.ndarray | None,
    minimum: np.ndarray,
    limit: np.ndarray,
) -> np.ndarray:
    """Alpha from a high-resolution matte of the edited object.

    The matte is used as is: its fractional values are never thresholded, blurred
    or multiplied by a binary mask, so fur and hair keep their detail. Two guards
    only decide which *objects* of the matte count:

    - ``limit``: a rectangle, so it cannot cut strands.
    - Connectivity: matte objects that do not touch ``minimum`` (neighbors,
      background things the matte also found) are dropped as a whole.

    ``residual`` (optional leakage refinement) lowers alpha only where Klein did
    not really change the pixel outside the original mask, e.g. a drifted
    background: those pixels fall back to the original.
    """
    alpha = matte * limit
    blobs, _ = label((alpha > 0.05) | minimum)
    keep = np.unique(blobs[minimum]) if minimum.any() else np.unique(blobs)
    alpha = alpha * np.isin(blobs, keep[keep != 0])
    if residual is not None:
        unchanged = np.clip(residual / (2 * MIN_CHANGE), 0.0, 1.0)
        alpha = np.where(minimum, alpha, alpha * unchanged)
    return alpha.astype(np.float32)


def change_alpha(
    before: Image.Image,
    after: Image.Image,
    minimum: np.ndarray,
    limit: np.ndarray,
    reach: float | None = None,
    debug: dict | None = None,
) -> np.ndarray:
    """Per-pixel paste weight in [0, 1]: how much Klein changed each pixel.

    There is no shape to guess. The change itself is the alpha, so fine
    structures such as fur strands, hair or leaves are pasted exactly where Klein
    recolored or redrew them, and untouched pixels stay original.

    1. ``delta`` = Klein minus original, per pixel in LAB.
    2. Klein also shifts the tone of the whole crop. Estimate that drift from
       pixels that clearly are not edited (low change, away from ``minimum``) as a
       smooth field and subtract it, so only the real edit remains.
    3. Otsu's threshold over ``limit`` separates edited from unedited residual;
       the residual is ramped into alpha around it.
    4. With ``minimum``: it is always 1, and pixels outside it fade out within
       ``reach`` px (the fur fringe). With no minimum, only blobs that are big
       enough are kept.
    """
    residual, drift_size = change_residual(before, after, minimum, limit, reach)
    has_minimum = bool(minimum.any())
    distance = distance_transform_edt(~minimum) if has_minimum else None
    values = residual[limit] if limit.any() else residual.ravel()
    otsu = float(threshold_otsu(values)) if np.ptp(values) > 1e-3 else MIN_CHANGE
    high = max(otsu, 2 * MIN_CHANGE)
    low = max(MIN_CHANGE, RAMP_START * high)
    alpha = np.clip((residual - low) / (high - low), 0.0, 1.0) * limit

    if has_minimum:
        if reach:
            alpha = alpha * np.clip(2.0 - distance / reach, 0.0, 1.0)
        # Keep only changes connected to the object.
        blobs, _ = label((alpha > 0.05) | minimum)
        keep = np.unique(blobs[minimum])
        alpha = np.where(minimum, 1.0, alpha * np.isin(blobs, keep[keep != 0]))
    else:
        blobs, _ = label(alpha > 0.05)
        sizes = np.bincount(blobs.ravel())
        keep = np.flatnonzero(sizes >= MIN_BLOB_PIXELS)
        alpha = alpha * np.isin(blobs, keep[keep != 0])
    alpha = np.maximum(alpha, binary_fill_holes(alpha > 0.5) & limit)
    if debug is not None:
        debug.update(
            diff=residual,
            low=low,
            high=high,
            minimum=minimum,
            limit=limit,
            ramp=alpha,
            drift=drift_size,
        )
    return alpha.astype(np.float32)


def blend_edit(
    before: Image.Image,
    after: Image.Image,
    *,
    minimum: np.ndarray,
    limit: np.ndarray,
    reach: float | None = None,
    predictor=None,
    matter=None,
    refine_leakage: bool = False,
    debug: dict | None = None,
) -> Image.Image:
    """Shared post-process of every Klein edit.

    1. Per-pixel alpha, one of:
       - ``matter`` given: ``matter(after)`` is a high-resolution float matte of
         the edited object (BiRefNet HR-Matting), used as the alpha so fur and
         hair keep their detail (see ``matte_alpha``). ``refine_leakage`` also
         reverts background that Klein only tone-shifted.
       - otherwise ``change_alpha``: weight of Klein's change beyond its global
         tone drift (object mask is the minimum; ``reach`` bounds the fringe).
    2. Harmonize ``after`` with PCT-Net on the confidently changed region.
    3. Composite ``after`` onto ``before`` with that alpha.
    """
    if matter is not None:
        matte = np.asarray(matter(after), dtype=np.float32)
        residual = (
            change_residual(before, after, minimum, limit, None)[0]
            if refine_leakage
            else None
        )
        alpha = matte_alpha(matte, residual, minimum, limit)
        if debug is not None:
            debug.update(matte=matte, minimum=minimum, limit=limit, ramp=alpha)
            if residual is not None:
                debug.update(diff=residual)
    else:
        alpha = change_alpha(before, after, minimum, limit, reach, debug)
    region = alpha > 0.5
    if not (alpha > 0.01).any():
        return before.copy()
    klein = after
    if predictor is not None and region.any():
        mask = Image.fromarray(region.astype(np.uint8) * 255, "L")
        after = harmonize_with_mask(predictor, after, mask)
    if debug is not None:
        debug.update(
            before=before,
            klein=klein,
            harmonized=after,
            alpha=(alpha * 255).astype(np.uint8),
            region=region,
        )
    return Image.composite(
        after, before, Image.fromarray((alpha * 255).astype(np.uint8), "L")
    )
