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
from framework.inference.flux2_klein import flux2_klein_edit_object, flux2_klein_remove_object
from framework.inference.sam3 import sam3_segment_mask
from framework.utils.randomness import seed_random_generators

DEFAULT_PROMPTS = [
    "Change the color of the object to blue",
]
DATASET_ID = "ziqiangoodgood/MIRAGE"
ANNOTATION_ROOT = Path(".out/annotation/mira_bench")


def latest_annotation_file() -> Path:
    """Resolve the MIRA annotation file, honouring the same env vars as the notebooks."""
    from framework.utils.project import load_project_environment

    load_project_environment()
    filename = "mira_bench_mixed.jsonl"
    explicit = os.environ.get("MIXED_PROMPT_MIRA_BENCH_ANNOTATIONS")
    if explicit:
        return Path(explicit).expanduser()
    root = Path(os.environ.get("ANNOTATE_OUT") or ANNOTATION_ROOT).expanduser()
    run = os.environ.get("MIXED_PROMPT_MIRA_BENCH_ANNOTATION_RUN") or "latest"
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
    for key in ("semantic_heat", "paste_weight"):
        if key in debug:
            debug[key].save(folder / f"{key}.png")
    if "refined" in debug:
        debug["refined"].save(folder / "after_klein_refine.png")
        debug["refined_raw"].save(folder / "klein_refine_raw.png")
    for key in ("mask_source", "attempt", "anchor"):
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
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=None, help="Override Klein steps")
    parser.add_argument(
        "--use-agent-flow",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Mimic agent execution up to edit_object (SAM segmentation + remove_object prior step)",
    )
    parser.add_argument(
        "--skip-remove",
        action="store_true",
        default=False,
        help="Opt out of the remove_object step to isolate whether remove_object caused the degradation",
    )
    args = parser.parse_args()
    prompts = args.prompt or DEFAULT_PROMPTS

    # Set RANDOM_SEED environment variable and initialize all RNGs exactly like agent eval
    os.environ["RANDOM_SEED"] = str(args.seed)
    seed_random_generators()

    annotation_file = args.annotation_file or latest_annotation_file()
    record = load_record(annotation_file, args.sample_id)
    ref_e1 = find_region(record, "e1")
    ref_e2 = find_region(record, "e2")

    box_0 = ref_e1["region"]["bbox_xyxy"]  # right cat: [620, 300, 917, 713]
    box_1 = ref_e2["region"]["bbox_xyxy"]  # left cat:  [75, 199, 301, 737]

    source = load_source_image(record)
    pool = ModelPool(InferenceConfig())
    if args.steps is not None:
        object.__setattr__(pool.config.klein, "num_inference_steps", args.steps)

    if args.use_agent_flow:
        print("=== Mimicking Real Agent Workflow ===")
        if args.skip_remove:
            print("[Agent Step 1] SKIPPED (--skip-remove)")
            current_image = source
        else:
            # Step 1: mask_0 = segment_object(image_0, box=box_0)
            print(f"[Agent Step 1a] segment_object box_0: {box_0}")
            mask_0 = sam3_segment_mask(pool, source, box=box_0)

            # Step 1: image_1 = remove_object(image_0, mask_0)
            print("[Agent Step 1b] remove_object(image_0, mask_0)")
            current_image = flux2_klein_remove_object(pool, source, mask_0)

        # Step 2: mask_1 = segment_object(image_1, box=box_1)
        print(f"[Agent Step 2a] segment_object box_1: {box_1}")
        mask_1 = sam3_segment_mask(pool, current_image, box=box_1)

        input_image = current_image
        edit_mask = mask_1
    else:
        print("=== Direct Isolated Edit Mode ===")
        mask_path = annotation_file.parent / ref_e2["region"]["mask_path"]
        with Image.open(mask_path) as mask_file:
            edit_mask = mask_file.convert("L")
        if edit_mask.size != source.size:
            edit_mask = edit_mask.resize(source.size, Image.Resampling.NEAREST)
        input_image = source

    outputs = []
    debugs = []
    for index, prompt in enumerate(prompts):
        start = time.time()
        debug: dict = {}
        output = flux2_klein_edit_object(
            pool,
            input_image,
            edit_mask,
            prompt,
            mode=args.mode,
            padding=args.padding,
            debug=debug,
        )
        print(f"[{index}] {prompt!r}: {time.time() - start:.1f}s")
        outputs.append(output)
        debugs.append(debug)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{args.sample_id.replace(':', '_')}_{args.entity_id}_{args.mode}"
    overlay = input_image.copy()
    tint = Image.new("RGB", input_image.size, (255, 0, 255))
    overlay.paste(tint, mask=edit_mask.point(lambda v: 90 if v >= 128 else 0))
    images = [source, input_image, overlay, *outputs]
    titles = ["source", "input", "mask", *(f"prompt {i}" for i in range(len(outputs)))]
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
