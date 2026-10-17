#!/usr/bin/env python3
"""Debug ``flux2_klein_remove_object`` on the remove step of a MIRA-Bench sample.

Loads the source image and the annotated mask of the entity being removed
(default: sample ``mirage:10.jpg``, entity ``e4``, the paper on the ground),
runs Klein inpaint removal, and saves a side-by-side comparison.

    uv run python -m scripts.test_remove_object
    uv run python -m scripts.test_remove_object --sample-id mirage:4.jpg --entity-id e3
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from framework.inference import InferenceConfig, ModelPool
from framework.inference.flux2_klein import flux2_klein_remove_object

DATASET_ID = "ziqiangoodgood/MIRAGE"
ANNOTATION_ROOT = Path(".out/annotation/mira_bench")


def latest_annotation_file() -> Path:
    runs = sorted(ANNOTATION_ROOT.glob("run_*/mira_bench_mixed.jsonl"))
    if not runs:
        raise FileNotFoundError(f"No annotation run found under {ANNOTATION_ROOT}")
    return runs[-1]


def load_record(annotation_file: Path, sample_id: str) -> dict:
    with annotation_file.open() as handle:
        for line in handle:
            record = json.loads(line)
            if record.get("sample_id") == sample_id:
                return record
    raise KeyError(f"{sample_id} not found in {annotation_file}")


def load_source_image(record: dict) -> Image.Image:
    from huggingface_hub import snapshot_download

    snapshot = snapshot_download(
        repo_id=DATASET_ID,
        repo_type="dataset",
        allow_patterns=["benchmark/*.jpg", "benchmark/*.jpeg", "benchmark/*.png"],
    )
    reference = record["source_ref"]
    prefix = next(
        p
        for p in ("mira_bench://benchmark/", "hf://ziqiangoodgood/MIRAGE/benchmark/")
        if reference.startswith(p)
    )
    with Image.open(
        Path(snapshot) / "benchmark" / reference.removeprefix(prefix)
    ) as image:
        return image.convert("RGB")


def find_region(record: dict, entity_id: str | None) -> dict:
    references = record["references"]
    if entity_id is None:
        # The remove clause is the reference whose span follows the word "remove".
        instruction = record["original_instruction"]
        for reference in references:
            if instruction[: reference["start"]].rstrip().lower().endswith("remove"):
                return reference
        raise ValueError("No 'remove' reference found; pass --entity-id")
    for reference in references:
        if reference["entity_id"] == entity_id:
            return reference
    raise KeyError(f"entity {entity_id} not found")


def labelled(image: Image.Image, text: str) -> Image.Image:
    image = image.copy()
    ImageDraw.Draw(image).text((8, 8), text, fill=(255, 0, 0))
    return image


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-id", default="mirage:10.jpg")
    parser.add_argument("--entity-id", default=None, help="Default: the removed entity")
    parser.add_argument("--annotation-file", type=Path, default=None)
    parser.add_argument(
        "--output-dir", type=Path, default=Path(".out/debug/remove_object")
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--mask-padding-ratio", type=float, default=0.05)
    parser.add_argument("--context-padding", type=float, default=1.5)
    parser.add_argument("--feather-px", type=int, default=24)
    parser.add_argument("--steps", type=int, default=None, help="Override Klein steps")
    args = parser.parse_args()

    annotation_file = args.annotation_file or latest_annotation_file()
    record = load_record(annotation_file, args.sample_id)
    reference = find_region(record, args.entity_id)
    mask_path = annotation_file.parent / reference["region"]["mask_path"]
    print(f"annotation: {annotation_file}")
    print(f"removing {reference['entity_id']}: {reference['span']!r} ({mask_path})")

    source = load_source_image(record)
    with Image.open(mask_path) as mask_file:
        mask = mask_file.convert("L")
    if mask.size != source.size:
        mask = mask.resize(source.size, Image.Resampling.NEAREST)

    pool = ModelPool(InferenceConfig())
    if args.steps is not None:
        object.__setattr__(pool.config.klein, "num_inference_steps", args.steps)

    start = time.time()
    output = flux2_klein_remove_object(
        pool,
        source,
        mask,
        seed=args.seed,
        mask_padding_ratio=args.mask_padding_ratio,
        context_padding=args.context_padding,
        feather_px=args.feather_px,
    )
    print(f"removal took {time.time() - start:.1f}s")

    changed = np.abs(
        np.asarray(output, dtype=np.int16) - np.asarray(source, dtype=np.int16)
    ).max(axis=2)
    mask_np = np.asarray(mask) >= 128
    print(
        f"mean change inside mask: {changed[mask_np].mean():.1f}; "
        f"pixels changed >8 outside mask: {int((changed[~mask_np] > 8).sum())}"
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = (
        f"{args.sample_id.replace(':', '_')}_{reference['entity_id']}_seed{args.seed}"
    )
    overlay = source.copy()
    tint = Image.new("RGB", source.size, (255, 0, 255))
    overlay.paste(tint, mask=mask.point(lambda v: 90 if v >= 128 else 0))
    panel = Image.new("RGB", (source.width * 3, source.height))
    for index, (image, title) in enumerate(
        ((source, "source"), (overlay, "mask"), (output, "removed"))
    ):
        panel.paste(labelled(image, title), (index * source.width, 0))
    panel.save(args.output_dir / f"{stem}_comparison.png")
    output.save(args.output_dir / f"{stem}_output.png")
    print(f"saved to {args.output_dir}/{stem}_*.png")


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
