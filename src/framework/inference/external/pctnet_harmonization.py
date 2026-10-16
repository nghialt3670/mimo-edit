"""Lightweight PCT-Net image harmonization (PIL API, no libcom).

Install Rakuten's PCT-Net repository under:
    src/framework/inference/external/pctnet_upstream/

Usage with an existing object mask:
    predictor = load_pctnet(device="cpu")
    result = harmonize_pctnet(predictor, generated_image, object_mask)

Usage with only a bounding box and before/after images:
    result, mask = harmonize_pctnet_with_box(
        predictor, original_image, generated_image, (x1, y1, x2, y2),
        return_mask=True,
    )

The before/after difference estimates which pixels belong to the inserted object.
This heuristic may fail if the generator also changes background pixels heavily.
PCT-Net adjusts foreground color/tonality, not geometry or texture.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence
import sys

import cv2
import numpy as np
import torch
from PIL import Image


DEFAULT_UPSTREAM = Path(__file__).resolve().parent / "pctnet_upstream"


def load_pctnet(
    *,
    device: str = "cpu",
    variant: str = "CNN",
    upstream_dir: str | Path = DEFAULT_UPSTREAM,
):
    """Load an official pretrained PCT-Net Predictor (reuse across calls)."""
    upstream_dir = Path(upstream_dir).expanduser().resolve()
    if variant not in ("CNN", "ViT"):
        raise ValueError("variant must be 'CNN' or 'ViT'")
    if not (upstream_dir / "iharm").is_dir():
        raise FileNotFoundError(
            f"PCT-Net source not found: {upstream_dir}. "
            "Clone the official repository there first."
        )

    weights = upstream_dir / "pretrained_models" / f"PCTNet_{variant}.pth"
    if not weights.is_file():
        raise FileNotFoundError(f"Missing PCT-Net checkpoint: {weights}")

    if str(upstream_dir) not in sys.path:
        sys.path.insert(0, str(upstream_dir))

    from iharm.inference.predictor import Predictor
    from iharm.inference.utils import load_model
    from iharm.mconfigs.base import BMCONFIGS

    config = BMCONFIGS[f"{variant}_pct"]["params"]
    model = load_model(f"{variant}_pct", str(weights), verbose=False)
    return Predictor(
        model,
        torch.device(device),
        use_attn=config["use_attn"],
        mean=config["input_normalization"]["mean"],
        std=config["input_normalization"]["std"],
    )


def harmonize_pctnet(
    predictor,
    image: Image.Image,
    mask: Image.Image,
) -> Image.Image:
    """Harmonize the masked region, preserving pixels outside it exactly."""
    if not isinstance(image, Image.Image) or not isinstance(mask, Image.Image):
        raise TypeError("image and mask must be PIL Images")
    if image.size != mask.size:
        raise ValueError("image and mask must have identical dimensions")

    rgb = image.convert("RGB")
    mask_l = mask.convert("L")
    # Support both binary 0/1 and grayscale 0..255 SAM outputs.
    if mask_l.getextrema()[1] <= 1:
        mask_l = mask_l.point(lambda v: v * 255)
    mask_l = mask_l.point(lambda v: 255 if v >= 128 else 0)
    if mask_l.getbbox() is None:
        return rgb.copy()

    rgb_np = np.asarray(rgb).copy()
    mask_np = np.asarray(mask_l).astype(np.float32) / 255.0
    rgb_small = cv2.resize(rgb_np, (256, 256), interpolation=cv2.INTER_LINEAR)
    mask_small = cv2.resize(mask_np, (256, 256), interpolation=cv2.INTER_LINEAR)

    _, output_np = predictor.predict(rgb_small, rgb_np, mask_small, mask_np)
    output_np = np.asarray(output_np)
    if output_np.shape != rgb_np.shape:
        raise RuntimeError(f"Unexpected PCT-Net output shape: {output_np.shape}")
    output_np = np.clip(np.nan_to_num(output_np), 0, 255).round().astype(np.uint8)
    harmonized = Image.fromarray(output_np, mode="RGB")

    # Even if the model predicts new background colors, don't apply them.
    return Image.composite(harmonized, rgb, mask_l)


def object_mask_from_before_after_box(
    original: Image.Image,
    generated: Image.Image,
    box: Sequence[float],
    *,
    threshold: int = 20,
    dilate_px: int = 1,
    keep_largest: bool = True,
) -> Image.Image:
    """Estimate a generated-object mask from pixel differences inside an XYXY box.

    White pixels identify the region to harmonize. Returns an L-mode image
    matching the full input resolution. No pixels outside the box are selected.

    This is a heuristic: changed background inside the box can be mistaken for
    foreground, especially when the generator edits the surrounding scene.
    """
    if not isinstance(original, Image.Image) or not isinstance(generated, Image.Image):
        raise TypeError("original and generated must be PIL Images")
    if original.size != generated.size:
        raise ValueError("original and generated must have identical dimensions")
    if len(box) != 4 or not all(np.isfinite(float(v)) for v in box):
        raise ValueError("box must contain four finite XYXY coordinates")
    if not (0 <= threshold <= 255):
        raise ValueError("threshold must be between 0 and 255")
    if dilate_px < 0:
        raise ValueError("dilate_px must be >= 0")

    width, height = original.size
    x1, y1, x2, y2 = (int(round(float(v))) for v in box)
    x1, x2 = max(0, x1), min(width, x2)
    y1, y2 = max(0, y1), min(height, y2)
    if x1 >= x2 or y1 >= y2:
        raise ValueError("box must intersect the image with positive area")

    src = np.asarray(original.convert("RGB"))[y1:y2, x1:x2]
    dst = np.asarray(generated.convert("RGB"))[y1:y2, x1:x2]
    diff = cv2.absdiff(src, dst)
    diff_gray = cv2.cvtColor(diff, cv2.COLOR_RGB2GRAY)
    _, binary = cv2.threshold(diff_gray, threshold, 255, cv2.THRESH_BINARY)

    # Join tiny gaps in a changed object's silhouette.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    if keep_largest and np.any(binary):
        count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
        if count > 1:
            largest_id = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
            binary = (labels == largest_id).astype(np.uint8) * 255

    if dilate_px and np.any(binary):
        diameter = 2 * dilate_px + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (diameter, diameter))
        binary = cv2.dilate(binary, kernel, iterations=1)

    full_mask = np.zeros((height, width), dtype=np.uint8)
    full_mask[y1:y2, x1:x2] = binary
    return Image.fromarray(full_mask, mode="L")


def harmonize_pctnet_with_box(
    predictor,
    original: Image.Image,
    generated: Image.Image,
    box: Sequence[float],
    *,
    threshold: int = 20,
    dilate_px: int = 1,
    keep_largest: bool = True,
    return_mask: bool = False,
) -> Image.Image | tuple[Image.Image, Image.Image]:
    """Harmonize only pixels changed by insertion within the given bounding box.

    `original` is the image BEFORE insertion; `generated` is the image AFTER.
    Unselected pixels are restored from `generated`, not from `original`.
    Set `return_mask=True` to inspect the automatically estimated object mask.
    """
    mask = object_mask_from_before_after_box(
        original,
        generated,
        box,
        threshold=threshold,
        dilate_px=dilate_px,
        keep_largest=keep_largest,
    )
    result = harmonize_pctnet(predictor, generated, mask)
    return (result, mask) if return_mask else result
