from __future__ import annotations

import json
import math
from collections import defaultdict
from collections.abc import Sequence
from statistics import fmean
from typing import Any

from PIL import Image, ImageDraw


def decode_refedit_mask(
    value: Any,
    *,
    size: tuple[int, int],
) -> Image.Image:
    """Decode RefEdit's polygon-serialized GT mask into a binary PIL image."""
    if isinstance(value, Image.Image):
        mask = value.convert("L")
        if mask.size != size:
            raise ValueError(
                f"RefEdit mask size {mask.size} does not match source image size {size}"
            )
        return mask.point(lambda pixel: 255 if pixel > 0 else 0)

    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError("RefEdit mask string is not valid JSON") from exc

    if not isinstance(value, Sequence) or isinstance(value, (bytes, bytearray)):
        raise TypeError(
            f"Unsupported RefEdit mask value: {type(value).__name__}; expected polygons"
        )

    polygons: Sequence[Any] = value
    if polygons and isinstance(polygons[0], (int, float)):
        polygons = [polygons]

    mask = Image.new("L", size, 0)
    draw = ImageDraw.Draw(mask)
    for polygon in polygons:
        if not isinstance(polygon, Sequence) or isinstance(
            polygon, (str, bytes, bytearray)
        ):
            raise ValueError("RefEdit polygon must be a sequence of x/y coordinates")
        if len(polygon) < 6 or len(polygon) % 2:
            raise ValueError(
                "RefEdit polygon must contain at least three x/y coordinate pairs"
            )
        coordinates = [float(coordinate) for coordinate in polygon]
        if not all(math.isfinite(coordinate) for coordinate in coordinates):
            raise ValueError("RefEdit polygon coordinates must be finite numbers")
        points = list(zip(coordinates[::2], coordinates[1::2], strict=True))
        draw.polygon(points, fill=255)
    return mask


def crop_pair_to_mask_roi(
    source: Image.Image,
    edited: Image.Image,
    mask_value: Any,
) -> tuple[Image.Image, Image.Image, tuple[int, int, int, int]]:
    """Crop source and edited images to the exact tight bounding box of a GT mask."""
    source_rgb = source.convert("RGB")
    edited_rgb = edited.convert("RGB")
    if edited_rgb.size != source_rgb.size:
        edited_rgb = edited_rgb.resize(source_rgb.size, Image.Resampling.LANCZOS)

    mask = decode_refedit_mask(mask_value, size=source_rgb.size)
    box = mask.getbbox()
    if box is None:
        raise ValueError("RefEdit GT mask is empty; cannot extract the ROI")
    return source_rgb.crop(box), edited_rgb.crop(box), box


def summarize_viescore_scores(
    rows: Sequence[dict[str, Any]],
) -> dict[str, dict[str, float | int]]:
    """Return mean SC, PQ, and O scores grouped by normalized category."""
    grouped: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: {"sc": [], "pq": [], "o": []}
    )
    for row in rows:
        category = str(row["category"]).strip().lower()
        if category not in {"easy", "hard"}:
            raise ValueError(f"Unexpected RefEdit category: {row['category']!r}")
        for metric in ("sc", "pq", "o"):
            score = float(row[metric])
            if not math.isfinite(score):
                raise ValueError(f"Non-finite VIEScore {metric}: {score}")
            grouped[category][metric].append(score)

    return {
        category: {
            **{metric: fmean(values) for metric, values in metrics.items()},
            "count": len(metrics["sc"]),
        }
        for category, metrics in sorted(grouped.items())
    }
