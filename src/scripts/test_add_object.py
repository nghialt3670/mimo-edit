#!/usr/bin/env python3
"""Run hard-coded add-object cases on the four MIRA-Bench samples.

Each case is a sample, a target (an annotated entity or a hand-picked box), a
prompt, and optionally a removal first (replace = remove + add). Run any subset:

    uv run python -m scripts.test_add_object --list
    uv run python -m scripts.test_add_object m10_dog_on_right_bench
    uv run python -m scripts.test_add_object m30_roses_on_doormat m57_dog_replace_rope --how both
    uv run python -m scripts.test_add_object --all

Output goes to ``.out/debug/add_object/``: the result, a comparison panel and a
``_debug`` folder per run (before, Klein output, matte, composite before
harmonization, harmonized crop).
"""

from __future__ import annotations

import argparse
import os
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image, ImageDraw

from framework.inference import InferenceConfig, ModelPool
from framework.inference.flux2_klein import (
    flux2_klein_add_object_with_box,
    flux2_klein_add_object_with_mask,
    flux2_klein_remove_object,
)
from framework.inference.flux2_klein.utils import NothingAdded

from .test_edit_object import (
    find_region,
    latest_annotation_file,
    load_record,
    load_source_image,
    save_debug,
)


@dataclass(frozen=True)
class Case:
    sample_id: str
    prompt: str
    entity_id: str | None = None
    # XYXY in the annotation's 1024-pixel space; used when there is no entity.
    box: tuple[int, int, int, int] | None = None
    remove_first: bool = False
    note: str = ""


CASES: dict[str, Case] = {
    "m4_dog_on_floor": Case(
        "mirage:4.jpg",
        "a small dog",
        box=(300, 800, 620, 960),
        note="no add in the annotation: hand-picked empty floor in front of the buckets",
    ),
    "m10_dog_on_right_bench": Case(
        "mirage:10.jpg", "a small dog", entity_id="e3", note="add a small dog onto v3"
    ),
    "m10_sticker_on_box": Case(
        "mirage:10.jpg",
        "a yellow sticker",
        entity_id="e5",
        note="add a yellow sticker onto v5",
    ),
    "m30_roses_on_doormat": Case(
        "mirage:30.jpg",
        "a bunch of roses",
        entity_id="e4",
        note="add a bunch of roses onto v4",
    ),
    "m30_dog_replace_middle_cat": Case(
        "mirage:30.jpg",
        "a dog",
        entity_id="e3",
        remove_first=True,
        note="replace v3 with a dog",
    ),
    "m57_dragon_on_lifebuoy": Case(
        "mirage:57.jpg",
        "a dragon pattern",
        entity_id="e4",
        note="add a dragon pattern onto v4",
    ),
    "m57_dog_replace_rope": Case(
        "mirage:57.jpg",
        "a lying dog",
        entity_id="e5",
        remove_first=True,
        note="replace v5 with a lying dog (touches two image edges)",
    ),
}


def labelled(image: Image.Image, text: str) -> Image.Image:
    image = image.copy()
    ImageDraw.Draw(image).text((8, 8), text, fill=(255, 0, 0))
    return image


def scaled_box(box, size, reference_size=(1024, 1024)) -> tuple[int, int, int, int]:
    sx, sy = size[0] / reference_size[0], size[1] / reference_size[1]
    x1, y1, x2, y2 = box
    return (round(x1 * sx), round(y1 * sy), round(x2 * sx), round(y2 * sy))


def run_case(
    name: str, case: Case, pool: ModelPool, hows: list[str], args: argparse.Namespace
) -> None:
    annotation_file = args.annotation_file or latest_annotation_file()
    record = load_record(annotation_file, case.sample_id)
    source = load_source_image(record)

    if case.entity_id is not None:
        region = find_region(record, case.entity_id)["region"]
        with Image.open(annotation_file.parent / region["mask_path"]) as handle:
            mask = handle.convert("L")
        if mask.size != source.size:
            mask = mask.resize(source.size, Image.Resampling.NEAREST)
        box = scaled_box(region["bbox_xyxy"], source.size, region["image_size"])
    else:
        assert case.box is not None
        box = scaled_box(case.box, source.size)
        mask = Image.new("L", source.size, 0)
        mask.paste(255, box)

    base = source
    if case.remove_first:
        start = time.time()
        base = flux2_klein_remove_object(pool, source, mask)
        print(f"  removed in {time.time() - start:.1f}s")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for how in hows:
        torch.manual_seed(args.seed)
        debug: dict = {}
        start = time.time()
        try:
            if how == "box":
                output = flux2_klein_add_object_with_box(
                    pool, base, box, case.prompt, debug=debug
                )
            else:
                output = flux2_klein_add_object_with_mask(
                    pool, base, mask, case.prompt, debug=debug
                )
        except NothingAdded as exc:
            print(f"  [{how}] FAILED: {exc}")
            continue
        print(f"  [{how}] {case.prompt!r}: {time.time() - start:.1f}s")

        stem = f"{name}_{how}"
        overlay = source.copy()
        overlay.paste(
            Image.new("RGB", source.size, (255, 0, 255)),
            mask=mask.point(lambda v: 90 if v >= 128 else 0),
        )
        images = [source, overlay, *([base] if case.remove_first else []), output]
        titles = ["source", "target", *(["removed"] if case.remove_first else []), how]
        panel = Image.new("RGB", (source.width * len(images), source.height))
        for index, (image, title) in enumerate(zip(images, titles)):
            panel.paste(labelled(image, title), (index * source.width, 0))
        panel.save(args.output_dir / f"{stem}_comparison.png")
        output.save(args.output_dir / f"{stem}.png")
        save_debug(debug, args.output_dir / f"{stem}_debug")
        print(f"  saved {args.output_dir / stem}*")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cases", nargs="*", help="case names (see --list)")
    parser.add_argument("--list", action="store_true", help="list the cases and exit")
    parser.add_argument("--all", action="store_true", help="run every case")
    parser.add_argument(
        "--how",
        default="box",
        choices=["box", "mask", "both"],
        help="which add function to call",
    )
    parser.add_argument("--annotation-file", type=Path, default=None)
    parser.add_argument(
        "--output-dir", type=Path, default=Path(".out/debug/add_object")
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.list:
        for name, case in CASES.items():
            kind = "remove+add" if case.remove_first else "add"
            print(f"{name:28s} {case.sample_id:14s} {kind:11s} {case.note}")
        return
    names = list(CASES) if args.all else args.cases
    if not names:
        parser.error("name at least one case, or use --all / --list")
    unknown = [name for name in names if name not in CASES]
    if unknown:
        parser.error(f"unknown case(s): {unknown}; see --list")

    hows = ["box", "mask"] if args.how == "both" else [args.how]
    pool = ModelPool(InferenceConfig())
    for name in names:
        print(f"{name}: {CASES[name].note}")
        run_case(name, CASES[name], pool, hows, args)


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
