from __future__ import annotations

import json
from pathlib import Path
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
        raise TypeError(
            f"RefEdit-Bench mask must be a polygon string, got {type(value).__name__}"
        )

    polygons = json.loads(value)
    if not isinstance(polygons, list):
        raise ValueError("RefEdit-Bench mask must contain a list of polygons")

    mask = Image.new("L", size, 0)
    draw = ImageDraw.Draw(mask)
    for polygon in polygons:
        if not isinstance(polygon, list) or len(polygon) < 6 or len(polygon) % 2:
            raise ValueError(
                "RefEdit-Bench polygon must contain at least three x/y coordinate pairs"
            )
        points = list(zip(polygon[::2], polygon[1::2], strict=True))
        draw.polygon(points, fill=255)
    return mask


def refedit_bench(row: dict[str, Any], split: str, index: int) -> Sample:
    """Official bpathir1/RefEdit-Bench schema (HF test split)."""
    required = {
        "input_image",
        "editing_instruction",
        "edit_instruction_single",
        "mask",
        "category",
    }
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
    required = {
        "img_id",
        "turn_index",
        "source_img",
        "mask_img",
        "instruction",
        "target_img",
    }
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
    required = {
        "task",
        "image_path",
        "input_image",
        "edited_image",
        "instruction",
        "caption",
        "mask",
    }
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


def _mira_bench_masks(
    polygons: Any,
    size: tuple[int, int],
    expected_count: int,
) -> tuple[Image.Image, list[Image.Image]]:
    if not isinstance(polygons, list) or len(polygons) != expected_count:
        raise ValueError(
            f"MIRA-Bench mask must contain {expected_count} polygons, "
            f"got {len(polygons) if isinstance(polygons, list) else type(polygons).__name__}"
        )

    union = Image.new("L", size, 0)
    union_draw = ImageDraw.Draw(union)
    masks: list[Image.Image] = []
    for index, polygon in enumerate(polygons):
        if (
            isinstance(polygon, list)
            and len(polygon) == 1
            and isinstance(polygon[0], list)
        ):
            polygon = polygon[0]
        if not isinstance(polygon, list) or len(polygon) < 6 or len(polygon) % 2:
            raise ValueError(
                f"MIRA-Bench mask polygon {index} must contain at least three x/y pairs"
            )
        points = list(zip(polygon[::2], polygon[1::2], strict=True))
        mask = Image.new("L", size, 0)
        ImageDraw.Draw(mask).polygon(points, fill=255)
        if mask.getbbox() is None:
            raise ValueError(f"MIRA-Bench mask polygon {index} is empty")
        union_draw.polygon(points, fill=255)
        masks.append(mask)
    return union, masks


def mira_bench(
    row: dict[str, Any],
    split: str,
    index: int,
    *,
    image_root: str | Path,
) -> Sample:
    """Adapt MIRA-Bench JSONL rows and their per-reference polygons."""
    required = {"image", "editing_instruction", "mask"}
    _require(row, required, "MIRA-Bench")
    image_name = row["image"]
    if not isinstance(image_name, str) or not image_name:
        raise TypeError("MIRA-Bench image must be a non-empty relative path")

    root = Path(image_root).expanduser().resolve()
    relative_image_path = Path(image_name)
    if relative_image_path.is_absolute() or ".." in relative_image_path.parts:
        raise ValueError(
            f"MIRA-Bench image path escapes benchmark directory: {image_name!r}"
        )
    image_path = root / relative_image_path
    with Image.open(image_path) as image_file:
        image = image_file.convert("RGB")

    instruction = row["editing_instruction"]
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError("MIRA-Bench editing_instruction must be a non-empty string")
    union_mask, reference_masks = _mira_bench_masks(
        row["mask"],
        image.size,
        expected_count=5,
    )
    raw_index = int(row.get("_row_index", index))
    image_uri = f"hf://ziqiangoodgood/MIRAGE/benchmark/{image_name}"
    # Preserve the established sample ID so existing annotation runs can resume.
    return Sample(
        sample_id=f"mirage:{image_name}",
        benchmark="MIRA-Bench",
        split=split,
        image=image,
        instruction=instruction,
        mask=union_mask,
        source_ref=image_uri,
        target_ref=None,
        metadata={
            "row_index": raw_index,
            "image": image_name,
            "mask_kind": "object_instance",
        },
        reference_masks={
            f"e{reference_index}": mask
            for reference_index, mask in enumerate(reference_masks, start=1)
        },
    )
