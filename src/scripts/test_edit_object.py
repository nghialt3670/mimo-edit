#!/usr/bin/env python3
"""Debug ``flux2_klein_edit_object`` on a MIRA-Bench sample.

Loads the source image and the annotated mask of one entity (default: sample
``mirage:30.jpg``, entity ``e2``, the left cat, recolored to blue), runs the edit
with several prompt wordings and saves a side-by-side comparison.

    uv run python -m scripts.test_edit_object
    uv run python -m scripts.test_edit_object --entity-id e5 --prompt "Make it stone"
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from framework.inference import InferenceConfig, ModelPool
from framework.inference.flux2_klein import flux2_klein_edit_object

DEFAULT_PROMPTS = [
    "Change the color of the left cat to blue",
]
DATASET_ID = "ziqiangoodgood/MIRAGE"
ANNOTATION_ROOT = Path(".out/annotation/mira_bench")


def latest_annotation_file() -> Path:
    """Resolve the MIRA annotation file, honouring the same env vars as the notebooks."""
    from framework.utils.project import load_project_environment

    load_project_environment()
    filename = "mira_bench_mixed.jsonl"
    explicit = os.environ.get("MMEDIT_MIRA_BENCH_ANNOTATIONS")
    if explicit:
        return Path(explicit).expanduser()
    root = Path(os.environ.get("ANNOTATE_OUT") or ANNOTATION_ROOT).expanduser()
    run = os.environ.get("MMEDIT_MIRA_BENCH_ANNOTATION_RUN") or "latest"
    runs = sorted(root.glob(f"run_*/{filename}"))
    if run != "latest":
        runs = [r for r in runs if r.parent.name == run]
    if runs:
        return runs[-1]
    if (root / filename).is_file():
        return root / filename
    raise FileNotFoundError(f"No annotation run found under {root.resolve()}")


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


def find_region(record: dict, entity_id: str) -> dict:
    for reference in record["references"]:
        if reference["entity_id"] == entity_id:
            return reference
    raise KeyError(f"entity {entity_id} not found")


def labelled(image: Image.Image, text: str) -> Image.Image:
    image = image.copy()
    ImageDraw.Draw(image).text((8, 8), text, fill=(255, 0, 0))
    return image


def gray(array: np.ndarray) -> Image.Image:
    return Image.fromarray((array.astype(np.float32) * 255).astype(np.uint8), "L")


def save_debug(debug: dict, folder: Path) -> None:
    """Save the pipeline internals: crop, Klein output, seed mask, matte, and the harmonized/blended crops."""
    folder.mkdir(parents=True, exist_ok=True)
    if "matte" not in debug:
        print("no debug data (global mode?)")
        return
    debug["before"].save(folder / "before.png")
    debug["klein"].save(folder / "klein_raw.png")
    gray(debug["minimum"]).save(folder / "original_mask.png")
    gray(debug["matte"]).save(folder / "matte_raw.png")
    debug["harmonized"].save(folder / "after_harmonize.png")
    debug["composite"].save(folder / "composite_before_harmonize.png")
    if "shadow" in debug:
        debug["shadow"].save(folder / "shadow_darkening.png")
    for key in ("shadow_source", "mask_source", "attempt", "anchor"):
        if key in debug:
            print(f"  {key}: {debug[key]}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-id", default="mirage:30.jpg")
    parser.add_argument("--entity-id", default="e2")
    parser.add_argument(
        "--prompt",
        action="append",
        help="Repeatable; each prompt is run with the same seed and compared",
    )
    parser.add_argument(
        "--mode", default="local", choices=["local", "regional", "global"]
    )
    parser.add_argument("--padding", type=float, default=0.25)
    parser.add_argument("--annotation-file", type=Path, default=None)
    parser.add_argument(
        "--output-dir", type=Path, default=Path(".out/debug/edit_object")
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=None, help="Override Klein steps")
    args = parser.parse_args()
    prompts = args.prompt or DEFAULT_PROMPTS

    annotation_file = args.annotation_file or latest_annotation_file()
    record = load_record(annotation_file, args.sample_id)
    reference = find_region(record, args.entity_id)
    mask_path = annotation_file.parent / reference["region"]["mask_path"]
    print(f"editing {reference['entity_id']}: {reference['span']!r} ({mask_path})")

    source = load_source_image(record)
    with Image.open(mask_path) as mask_file:
        mask = mask_file.convert("L")
    if mask.size != source.size:
        mask = mask.resize(source.size, Image.Resampling.NEAREST)

    pool = ModelPool(InferenceConfig())
    if args.steps is not None:
        object.__setattr__(pool.config.klein, "num_inference_steps", args.steps)

    outputs = []
    debugs = []
    for index, prompt in enumerate(prompts):
        # Same seed for every prompt so only the wording differs.
        torch.manual_seed(args.seed)
        start = time.time()
        debug: dict = {}
        output = flux2_klein_edit_object(
            pool,
            source,
            mask,
            prompt,
            mode=args.mode,
            padding=args.padding,
            debug=debug,
        )
        print(f"[{index}] {prompt!r}: {time.time() - start:.1f}s")
        outputs.append(output)
        debugs.append(debug)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{args.sample_id.replace(':', '_')}_{reference['entity_id']}_{args.mode}"
    overlay = source.copy()
    tint = Image.new("RGB", source.size, (255, 0, 255))
    overlay.paste(tint, mask=mask.point(lambda v: 90 if v >= 128 else 0))
    images = [source, overlay, *outputs]
    titles = ["source", "mask", *(f"prompt {i}" for i in range(len(outputs)))]
    panel = Image.new("RGB", (source.width * len(images), source.height))
    for index, (image, title) in enumerate(zip(images, titles)):
        panel.paste(labelled(image, title), (index * source.width, 0))
    panel.save(args.output_dir / f"{stem}_comparison.png")
    for index, output in enumerate(outputs):
        output.save(args.output_dir / f"{stem}_prompt{index}.png")
        save_debug(debugs[index], args.output_dir / f"{stem}_prompt{index}_debug")
    print(f"saved to {args.output_dir}/{stem}_*.png")


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
