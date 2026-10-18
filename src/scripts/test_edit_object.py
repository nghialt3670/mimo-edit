#!/usr/bin/env python3
"""Debug ``flux2_klein_edit_object`` on a MIRA-Bench sample.

Loads the source image and the annotated mask of one entity (default: sample
``mirage:30.jpg``, entity ``e2``, the left cat), runs the edit
with several prompt wordings and saves a side-by-side comparison.

    uv run python -m scripts.test_edit_object
    uv run python -m scripts.test_edit_object --entity-id e2 --prompt "Make it stone"
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw

from framework.inference import InferenceConfig, ModelPool
from framework.inference.flux2_klein import flux2_klein_edit_object

DEFAULT_PROMPTS = [
    "Change the color of the cat to blue",
]
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
    """Save the paste-back internals: what Klein changed and what was pasted."""
    folder.mkdir(parents=True, exist_ok=True)
    if "ramp" not in debug:
        print("no debug data (global mode?)")
        return
    region, minimum = debug["region"], debug["minimum"]
    print(
        f"  minimum {int(minimum.sum())} px, pasted (alpha>0.5) {int(region.sum())} px "
        f"(+{int((region & ~minimum).sum())} beyond mask)"
    )
    if "matte" in debug:
        # The raw 8-bit matte straight from BiRefNet, before any guard.
        gray(debug["matte"]).save(folder / "matte_raw.png")
    if "diff" in debug:
        diff = debug["diff"]
        heat = np.clip(diff / max(float(diff.max()), 1e-6), 0, 1)
        colored = cv2.applyColorMap((heat * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
        Image.fromarray(colored[..., ::-1]).save(folder / "diff_heatmap.png")
    gray(debug["ramp"]).save(folder / "ramp_alpha.png")
    gray(debug["minimum"]).save(folder / "original_mask.png")
    gray(debug["limit"]).save(folder / "limit.png")
    gray(debug["region"]).save(folder / "final_region.png")
    Image.fromarray(debug["alpha"]).save(folder / "paste_alpha.png")
    debug["before"].save(folder / "before.png")
    debug["klein"].save(folder / "klein_raw.png")
    debug["harmonized"].save(folder / "after_harmonize.png")
    # Red: original mask; green: pasted beyond the mask.
    overlay = np.asarray(debug["klein"]).copy()
    overlay[minimum] = (overlay[minimum] * 0.5 + np.array([255, 0, 0]) * 0.5).astype(
        np.uint8
    )
    extra = region & ~minimum
    overlay[extra] = (overlay[extra] * 0.4 + np.array([0, 255, 0]) * 0.6).astype(
        np.uint8
    )
    Image.fromarray(overlay).save(folder / "region_on_klein.png")


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
    parser.add_argument("--no-harmonize", action="store_true")
    parser.add_argument(
        "--no-matting", action="store_true", help="Use the change-based alpha"
    )
    parser.add_argument(
        "--refine-leakage",
        action="store_true",
        help="Revert background that Klein only tone-shifted (matting only)",
    )
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
            harmonize=not args.no_harmonize,
            matting=not args.no_matting,
            refine_leakage=args.refine_leakage,
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
