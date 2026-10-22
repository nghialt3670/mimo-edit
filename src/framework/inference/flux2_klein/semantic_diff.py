"""LPIPS semantic difference between a crop and Klein's edit of it."""

from __future__ import annotations

import cv2
import numpy as np
import torch
from PIL import Image

MAX_SIDE = 448
THRESHOLD = 0.32
FEATHER = 10
_metric = None


def _lpips_metric(device: str):
    global _metric
    if _metric is None:
        import lpips

        _metric = lpips.LPIPS(net="alex", spatial=True, verbose=False).eval()
    return _metric.to(device)


def lpips_map(before: Image.Image, after: Image.Image) -> np.ndarray:
    """Per-pixel LPIPS-Alex distance at the size of ``before``."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    factor = min(1.0, MAX_SIDE / max(before.size))
    size = tuple(max(16, round(v * factor / 16) * 16) for v in before.size)

    def tensor(image: Image.Image):
        arr = np.asarray(image.resize(size, Image.Resampling.BICUBIC), np.float32)
        return torch.from_numpy(arr).permute(2, 0, 1)[None].to(device) / 127.5 - 1.0

    with torch.inference_mode():
        score = _lpips_metric(device)(tensor(before), tensor(after))[0, 0]
    return cv2.resize(
        score.float().cpu().numpy(), before.size, interpolation=cv2.INTER_CUBIC
    )


def semantic_diff_weight(
    before: Image.Image,
    after: Image.Image,
    *,
    threshold: float = THRESHOLD,
    feather: int = FEATHER,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(transfer_weight, normalized_heatmap)``, both float32 in [0, 1].

    The weight is 1 where the edit changed the crop semantically and fades out
    over ``feather`` pixels outside that region.
    """
    scores = cv2.GaussianBlur(np.maximum(lpips_map(before, after), 0.0), (0, 0), 1.5)
    low, high = np.quantile(scores, 0.05), np.quantile(scores, 0.995)
    if high - low < 1e-5:
        return np.zeros(scores.shape, np.float32), np.zeros(scores.shape, np.float32)
    norm = np.clip((scores - low) / (high - low), 0.0, 1.0).astype(np.float32)
    selected = norm >= threshold
    if not selected.any():
        return np.zeros(scores.shape, np.float32), norm
    distance = cv2.distanceTransform((~selected).astype(np.uint8), cv2.DIST_L2, 5)
    return np.clip(1.0 - distance / float(feather), 0.0, 1.0).astype(np.float32), norm
