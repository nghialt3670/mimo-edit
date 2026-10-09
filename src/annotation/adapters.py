from __future__ import annotations

import json
from typing import Any

from PIL import Image, ImageDraw

from .schema import Sample


def _require(row: dict[str, Any], required: set[str], dataset: str) -> None:
    missing = required - set(row)
    if missing:
        raise KeyError(f"{dataset}: missing {sorted(missing)}; available={sorted(row)}")


def _refedit_bench_mask(value: Any, size: tuple[int, int]) -> Image.Image:
    if isinstance(value, Image.Image):
        return value.convert("L")
    if not isinstance(value, str):
        raise TypeError(f"RefEdit-Bench mask must be a polygon string, got {type(value).__name__}")

    polygons = json.loads(value)
    if not isinstance(polygons, list):
        raise ValueError("RefEdit-Bench mask must contain a list of polygons")

    mask = Image.new("L", size, 0)
    draw = ImageDraw.Draw(mask)
    for polygon in polygons:
        if not isinstance(polygon, list) or len(polygon) < 6 or len(polygon) % 2:
            raise ValueError("RefEdit-Bench polygon must contain at least three x/y coordinate pairs")
        points = list(zip(polygon[::2], polygon[1::2], strict=True))
        draw.polygon(points, fill=255)
    return mask


def refedit_bench(row: dict[str, Any], split: str, index: int) -> Sample:
    """Official bpathir1/RefEdit-Bench schema (HF test split)."""
    required = {"input_image", "editing_instruction", "edit_instruction_single", "mask", "category"}
    _require(row, required, "RefEdit-Bench")
    category = str(row["category"])
    raw_index = int(row.get("_row_index", index))
    return Sample(
        sample_id=f"{category.lower()}:{raw_index:03d}",
        benchmark="refedit_bench",
        split=category.lower(),
        image=row["input_image"],
        instruction=str(row["editing_instruction"]),
        target=None,
        mask=_refedit_bench_mask(row["mask"], row["input_image"].size),
        source_ref=f"hf://bpathir1/RefEdit-Bench/{split}/{raw_index}/input_image",
        target_ref=None,
        metadata={
            "row_index": raw_index,
            "hf_split": split,
            "category": category,
            "edit_instruction_single": row["edit_instruction_single"],
            "mask_kind": "object_instance",
        },
    )


def magicbrush(row: dict[str, Any], split: str, index: int) -> Sample:
    """Official osunlp/MagicBrush HF train/dev schema."""
    required = {"img_id", "turn_index", "source_img", "mask_img", "instruction", "target_img"}
    _require(row, required, "MagicBrush")
    img_id, turn = str(row["img_id"]), int(row["turn_index"])
    raw_index = int(row.get("_row_index", index))
    return Sample(
        sample_id=f"{img_id}:{turn}",
        benchmark="MagicBrush",
        split=split,
        image=row["source_img"],
        instruction=str(row["instruction"]),
        target=row["target_img"],
        mask=row["mask_img"],
        source_ref=f"hf://osunlp/MagicBrush/{split}/{raw_index}/source_img",
        target_ref=f"hf://osunlp/MagicBrush/{split}/{raw_index}/target_img",
        metadata={
            "row_index": raw_index,
            "img_id": img_id,
            "turn_index": turn,
            "mask_kind": "edit_region",
        },
    )


def compbench(row: dict[str, Any], split: str, index: int) -> Sample:
    """Official BohanJia/CompBench schema for both train and multi_turn splits."""
    required = {"task", "image_path", "input_image", "edited_image", "instruction", "caption", "mask"}
    _require(row, required, "CompBench")
    task, image_path = str(row["task"]), str(row["image_path"])
    raw_index = int(row.get("_row_index", index))
    return Sample(
        sample_id=f"{task}:{image_path}",
        benchmark="CompBench",
        split=split,
        image=row["input_image"],
        instruction=str(row["instruction"]),
        target=row["edited_image"],
        mask=row["mask"],
        source_ref=f"hf://BohanJia/CompBench/{split}/{raw_index}/input_image",
        target_ref=f"hf://BohanJia/CompBench/{split}/{raw_index}/edited_image",
        metadata={
            "row_index": raw_index,
            "task": task,
            "image_path": image_path,
            "caption": row["caption"],
            "mask_kind": "edit_region",
        },
    )
