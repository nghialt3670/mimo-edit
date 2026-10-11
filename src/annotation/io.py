from __future__ import annotations

import json
import os
import re
from pathlib import Path

from PIL import Image

from .grounder import resize_mask
from .schema import AnnotationRecord, Sample


def append_jsonl(path: str | Path, obj: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False, default=str) + "\n")


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_") or "sample"


def save_reference_assets(
    sample: Sample, record: AnnotationRecord, output_jsonl: str | Path
) -> None:
    output_jsonl = Path(output_jsonl)
    root = (
        output_jsonl.parent
        / "assets"
        / output_jsonl.stem
        / _safe_name(sample.sample_id)
    )
    root.mkdir(parents=True, exist_ok=True)
    source = sample.image.convert("RGB")

    for ref in record.references:
        x0, y0, x1, y1 = ref.region.bbox_xyxy
        crop = source.crop((int(x0), int(y0), int(x1), int(y1)))
        crop_path = root / f"{ref.entity_id}_crop.png"
        crop.save(crop_path)
        ref.region.crop_path = os.path.relpath(crop_path, output_jsonl.parent)

        if ref.region.source.startswith("benchmark_mask") and sample.mask is not None:
            mask_path = root / f"{ref.entity_id}_mask.png"
            reference_mask = sample.reference_masks.get(ref.entity_id, sample.mask)
            resize_mask(reference_mask, source.size).save(mask_path)
            ref.region.mask_path = os.path.relpath(mask_path, output_jsonl.parent)
