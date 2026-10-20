"""Helpers shared by the FLUX.2 Klein editing, insertion, and removal functions."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch
from PIL import Image

from framework.utils.randomness import get_random_seed

from ..config import KleinInferenceConfig
from ..vitmatte import vitmatte_alpha

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
    seed_offset: int = 0,
) -> Image.Image:
    with torch.inference_mode():
        return (
            pipeline(
                image=image,
                prompt=prompt,
                num_inference_steps=config.num_inference_steps,
                guidance_scale=config.guidance_scale,
                generator=torch.Generator(device="cpu").manual_seed(
                    get_random_seed() + seed_offset
                ),
            )
            .images[0]
            .convert("RGB")
        )


# Shared settings for distilled FLUX.2 Klein 4B.
PADDING_FRAC = 0
WORK_SIZE = 1024


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


class NothingAdded(RuntimeError):
    """Klein's output has no new content, so there is nothing to paste."""


def _changed_map(before: Image.Image, after: Image.Image, threshold: int) -> np.ndarray:
    from scipy.ndimage import uniform_filter

    delta = np.abs(
        np.asarray(after, np.int16) - np.asarray(before.convert("RGB"), np.int16)
    ).max(-1)
    return uniform_filter(delta.astype(np.float32), size=5) > threshold


def _largest_changes(changed: np.ndarray, keep: float = 0.1) -> np.ndarray:
    """Close and fill the change map, keeping components of a real size."""
    from scipy.ndimage import binary_closing, binary_fill_holes, label

    closed = binary_fill_holes(binary_closing(changed, iterations=3))
    labels, count = label(closed)
    if count == 0:
        return closed
    sizes = np.bincount(labels.ravel())[1:]
    keep_ids = 1 + np.nonzero(sizes >= keep * sizes.max())[0]
    return np.isin(labels, keep_ids)


def segment_added_object(
    pool,
    before: Image.Image,
    after: Image.Image,
    region: np.ndarray,
    prompt: str | None = None,
    *,
    change_threshold: int = 20,
    min_changed: float = 0.4,
    min_area: float = 0.01,
    debug: dict | None = None,
) -> np.ndarray:
    """Mask of the object Klein added (bool, crop coordinates), never the whole box.

    ``region`` is where the object was requested; the object may extend past it,
    but its centre has to fall inside. Candidates come from SAM3 text masks of
    ``prompt`` on ``after``, kept only if they are mostly new content (differ
    from ``before``). Without a usable candidate the mask is the cleaned map of
    changed pixels instead. If almost nothing changed, ``NothingAdded`` is raised.
    """
    from scipy.ndimage import binary_dilation

    changed = _changed_map(before, after, change_threshold)
    if changed.mean() < 0.002:
        raise NothingAdded("Klein's output is unchanged; no object was added")
    ys, xs = np.nonzero(region)
    margin = max(8, int(0.25 * max(np.ptp(ys), np.ptp(xs) + 1)))
    allowed = binary_dilation(region, iterations=margin)
    total = float(region.size)
    selected = np.zeros_like(region, bool)
    if prompt and prompt.strip():
        from ..sam3 import sam3_segment_masks

        for candidate in sam3_segment_masks(pool, after, prompt, min_confidence=0.3):
            mask = np.asarray(candidate) > 127
            if mask.sum() < min_area * total:
                continue
            cy, cx = (int(v.mean()) for v in np.nonzero(mask))
            if not allowed[cy, cx] or float(changed[mask].mean()) < min_changed:
                continue
            selected |= mask
    if selected.any():
        source = "text"
        mask = selected
    else:
        source = "change"
        mask = _largest_changes(changed & allowed)
        if mask.sum() < min_area * total:
            raise NothingAdded("no added object found in Klein's output")
    if debug is not None:
        debug["mask_source"] = source
    return mask


def harmonize_by_alpha(predictor, image: Image.Image, alpha: np.ndarray) -> Image.Image:
    """Harmonize with strength proportional to ``alpha`` (weak at the edge, full inside).

    PCT-Net only takes a binary mask, so it runs on every pixel where alpha is
    not black, and its result is mixed with the input by alpha per pixel.
    """
    support = Image.fromarray((alpha > 0).astype(np.uint8) * 255, "L")
    harmonized = np.asarray(
        harmonize_with_mask(predictor, image, support), dtype=np.float32
    )
    original = np.asarray(image, dtype=np.float32)
    mixed = original + np.clip(alpha, 0.0, 1.0)[..., None] * (harmonized - original)
    return Image.fromarray(mixed.round().astype(np.uint8), "RGB")


def blend_edit(
    pool,
    scene: Image.Image,
    crop_box: tuple[int, int, int, int],
    after: Image.Image,
    mask: np.ndarray,
    *,
    harmonize: bool = True,
    shadow: bool = False,
    ground_shadow: bool = False,
    debug: dict | None = None,
) -> tuple[Image.Image, np.ndarray]:
    """Paste Klein's edit into the whole ``scene`` using the ViTMatte matte.

    ``after`` is Klein's edit of ``scene.crop(crop_box)`` and ``mask`` the object
    region in crop coordinates; it only seeds the trimap. The matte of ``after``
    is the paste weight (no change detection, clamp or threshold).

    The matted object is first composited onto the full scene. With ``harmonize``
    PCT-Net then runs on that whole image, wherever the matte is not black, with
    strength scaled by the matte, so it sees the real surroundings. With
    ``shadow`` the object's cast shadow is carried over first as a darkening of
    the original scene (``ground_shadow`` adds a synthetic one if Klein drew
    none); see ``transfer_shadow``. Returns the
    full image and the crop-sized matte. ``debug`` receives the crop-sized
    composite before harmonization (``composite``) and the harmonized crop
    (``harmonized``).
    """
    matte = vitmatte_alpha(pool, after, mask)
    weight = Image.fromarray((matte * 255).astype(np.uint8), "L")
    full = scene.convert("RGB").copy()
    if shadow:
        from .shadow_transfer import transfer_shadow

        shade, source = transfer_shadow(
            full.crop(crop_box), after, matte, ground=ground_shadow
        )
        if source != "none":
            darkened = np.asarray(full.crop(crop_box), np.float32) * shade[..., None]
            full.paste(Image.fromarray(darkened.round().astype(np.uint8)), crop_box[:2])
        if debug is not None:
            debug["shadow_source"] = source
            debug["shadow"] = Image.fromarray(
                ((1.0 - shade) * 255).clip(0, 255).astype(np.uint8), "L"
            )
    full.paste(Image.composite(after, full.crop(crop_box), weight), crop_box[:2])
    if debug is not None:
        debug["composite"] = full.crop(crop_box)
    if harmonize and (matte > 0).any():
        canvas = np.zeros((scene.height, scene.width), np.float32)
        left, top, right, bottom = crop_box
        canvas[top:bottom, left:right] = matte
        full = harmonize_by_alpha(pool.pctnet_pipeline, full, canvas)
    if debug is not None:
        debug["harmonized"] = full.crop(crop_box)
    return full, matte
