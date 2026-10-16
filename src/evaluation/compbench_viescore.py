from __future__ import annotations

import json
import math
import os
import tempfile
from collections import defaultdict
from pathlib import Path
from statistics import fmean
from typing import Any, Literal

from PIL import Image
from tqdm.auto import tqdm

from evaluation.common import HFImageLoader, read_jsonl
from evaluation.compbench_metrics import _load_mask, collect_saved_outputs
from evaluation.metrics import crop_pair_to_mask_roi
from evaluation.viescore import load_viescore_scorer

PromptMode = Literal["text", "mixed"]
MULTI_OBJECT_TASKS = {"multi_object_add", "multi_object_remove"}
REPLICATE_TASKS = {"multi_object_add", "multi_object_remove"}
VIESCORE_NAMES = ("sc", "pq", "o")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary_path = Path(stream.name)
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def summarize_group(group: list[dict[str, Any]]) -> dict[str, float | int]:
        scores: dict[str, float] = {}
        for metric in VIESCORE_NAMES:
            values = [float(row[metric]) for row in group]
            if not all(math.isfinite(value) for value in values):
                raise ValueError(f"Non-finite cached VIEScore {metric}: {values!r}")
            scores[metric] = fmean(values)
        return {
            **scores,
            "count": len(group),
        }

    summary: dict[str, Any] = {}
    if rows:
        summary["overall"] = summarize_group(rows)
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_task[str(row["task"])].append(row)
    summary["by_task"] = {
        task: summarize_group(group) for task, group in sorted(by_task.items())
    }

    if any("model" in row for row in rows):
        by_model: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            by_model[str(row["model"])].append(row)
        summary["by_model"] = {
            model: {
                "overall": summarize_group(group),
                "by_task": {
                    task: summarize_group([row for row in group if row["task"] == task])
                    for task in sorted({str(row["task"]) for row in group})
                },
            }
            for model, group in sorted(by_model.items())
        }
    return summary


def _score_crop_pair(
    scorer: Any,
    source: Image.Image,
    generated_path: Path,
    mask: Any,
    instruction: str,
) -> tuple[tuple[float, float, float], tuple[int, int, int, int]]:
    with Image.open(generated_path) as generated_file:
        source_roi, generated_roi, roi_box = crop_pair_to_mask_roi(
            source, generated_file.copy(), mask
        )
    scores = scorer.evaluate(
        [source_roi, generated_roi],
        instruction,
        extract_all_score=True,
    )
    if not isinstance(scores, (list, tuple)) or len(scores) != 3:
        raise ValueError(f"Expected VIEScore [SC, PQ, O], got {scores!r}")
    values = (float(scores[0]), float(scores[1]), float(scores[2]))
    if not all(math.isfinite(value) for value in values):
        raise ValueError(f"Non-finite VIEScore result: {values!r}")
    return values, roi_box


def _resolve_viescore(
    scorer: Any | None,
    viescore_root: Path | None,
    key_path: Path | None,
) -> Any:
    if scorer is not None:
        return scorer
    viescore_root = (
        viescore_root or Path(os.environ.get("VIESCORE_ROOT", "")).expanduser()
    )
    key_path = key_path or Path(os.environ.get("VIESCORE_KEY_PATH", "")).expanduser()
    if not str(viescore_root) or str(viescore_root) == ".":
        raise EnvironmentError(
            "Set VIESCORE_ROOT to the Step1X-Edit GEdit-Bench directory."
        )
    if not str(key_path) or str(key_path) == ".":
        raise EnvironmentError(
            "Set VIESCORE_KEY_PATH to the local OpenAI API key file."
        )
    return load_viescore_scorer(viescore_root, key_path)


def _cache_key(
    row: dict[str, Any],
    *,
    output_path: Path,
    instruction: str,
    prompt_mode: PromptMode | None,
    model: str | None = None,
) -> tuple[Any, ...]:
    output_stat = output_path.stat()
    return (
        model,
        row.get("sample_id"),
        str(output_path),
        output_stat.st_size,
        output_stat.st_mtime_ns,
        instruction,
        prompt_mode,
    )


def _load_cached_scores(metric_path: Path) -> dict[tuple[Any, ...], dict[str, Any]]:
    cached: dict[tuple[Any, ...], dict[str, Any]] = {}
    if metric_path.is_file():
        for row in read_jsonl(metric_path):
            cached[
                (
                    row.get("model"),
                    row.get("sample_id"),
                    row.get("output_path"),
                    row.get("output_size_bytes"),
                    row.get("output_mtime_ns"),
                    row.get("instruction"),
                    row.get("prompt_mode"),
                )
            ] = row
    return cached


def evaluate_saved_compbench_outputs(
    records: list[dict[str, Any]],
    *,
    output_dir: Path,
    image_loader: HFImageLoader,
    prompt_mode: PromptMode,
    scorer: Any | None = None,
    viescore_root: Path | None = None,
    key_path: Path | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Score only CompBench multi-object outputs from a saved inference run."""
    if prompt_mode not in ("text", "mixed"):
        raise ValueError(f"Unsupported prompt mode: {prompt_mode!r}")
    output_dir = output_dir.expanduser().resolve()
    saved_outputs = collect_saved_outputs(
        records,
        output_dir,
        prompt_mode=prompt_mode,
        task_filter=MULTI_OBJECT_TASKS,
    )
    if not saved_outputs:
        raise FileNotFoundError(
            f"No saved multi-object CompBench outputs found in {output_dir}"
        )
    scorer = _resolve_viescore(scorer, viescore_root, key_path)

    metric_path = output_dir / "metrics" / "modified_viescore.jsonl"
    cached = _load_cached_scores(metric_path)
    rows: list[dict[str, Any]] = []
    for saved in tqdm(saved_outputs, desc="CompBench VIEScore", unit="sample"):
        index = saved["record_position"]
        record = records[index]
        metadata = record.get("metadata") or {}
        task = metadata.get("task")
        if task not in MULTI_OBJECT_TASKS:
            raise ValueError(
                f"Unexpected CompBench multi-object task for "
                f"{record.get('sample_id')}: {task!r}"
            )
        instruction = record.get("original_instruction")
        if not isinstance(instruction, str) or not instruction.strip():
            raise ValueError(
                f"Missing original_instruction for {record.get('sample_id')}"
            )
        output_path: Path = saved["output_path"]
        cache_key = _cache_key(
            record,
            output_path=output_path,
            instruction=instruction,
            prompt_mode=prompt_mode,
        )
        if cache_key in cached:
            score_row = cached[cache_key]
        else:
            source_reference = image_loader.parse_reference(record.get("source_ref"))
            if source_reference is None:
                raise ValueError(f"Missing source_ref for {record.get('sample_id')}")
            dataset_id, split, row_index, image_field = source_reference
            source_row = image_loader.get_row(dataset_id, split, row_index)
            if (
                row_index != metadata.get("row_index")
                or source_row.get("task") != task
                or source_row.get("image_path") != metadata.get("image_path")
                or source_row.get("instruction") != instruction
            ):
                raise ValueError(
                    f"Dataset/annotation mismatch for CompBench sample "
                    f"{record.get('sample_id')}"
                )
            scores, roi_box = _score_crop_pair(
                scorer,
                image_loader.as_image(source_row[image_field]),
                output_path,
                _load_mask(source_row["mask"]),
                instruction,
            )
            output_stat = output_path.stat()
            score_row = {
                "sample_id": record.get("sample_id"),
                "record_position": index,
                "dataset_row_index": metadata.get("row_index"),
                "task": task,
                "prompt_mode": prompt_mode,
                "instruction": instruction,
                "output_path": str(output_path),
                "output_size_bytes": output_stat.st_size,
                "output_mtime_ns": output_stat.st_mtime_ns,
                "roi_bbox_xyxy": list(roi_box),
                "sc": scores[0],
                "pq": scores[1],
                "o": scores[2],
            }
        rows.append(score_row)
        _write_jsonl(metric_path, rows)

    _write_jsonl(metric_path, rows)
    summary = _summarize(rows)
    summary_path = output_dir / "metrics" / "modified_viescore_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Scored {len(rows)} saved CompBench multi-object outputs.")
    print(f"Per-sample VIEScore: {metric_path}")
    print(f"Task summary: {summary_path}")
    return rows, summary


def _resolve_under_root(root: Path, relative_path: Any, field: str) -> Path:
    if not isinstance(relative_path, str) or not relative_path:
        raise ValueError(f"Missing {field} in Replicate CompBench record")
    path = (root / relative_path).resolve()
    if path != root and root not in path.parents:
        raise ValueError(f"Replicate {field} escapes result root: {relative_path!r}")
    if not path.is_file():
        raise FileNotFoundError(f"Replicate {field} not found: {path}")
    return path


def collect_replicate_saved_outputs(root: Path) -> list[dict[str, Any]]:
    root = root.expanduser().resolve()
    selection_path = root / "selection.jsonl"
    if not selection_path.is_file():
        raise FileNotFoundError(
            f"Replicate selection manifest not found: {selection_path}"
        )
    selection_by_id: dict[str, dict[str, Any]] = {}
    for sample in read_jsonl(selection_path):
        sample_id = sample.get("id")
        if not isinstance(sample_id, str) or not sample_id:
            raise ValueError(f"Missing id in {selection_path}")
        if sample_id in selection_by_id:
            raise ValueError(f"Duplicate sample id in {selection_path}: {sample_id!r}")
        selection_by_id[sample_id] = sample

    outputs = []
    seen: set[tuple[str, str]] = set()
    records_root = root / "records"
    for record_path in sorted(records_root.glob("*/*/*.json")):
        try:
            record = json.loads(record_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON record: {record_path}") from exc
        if not isinstance(record, dict):
            raise ValueError(f"Expected a JSON object in {record_path}")
        if record.get("status") != "succeeded":
            continue
        model = record.get("model")
        task = record.get("task")
        sample_id = record.get("sample_id")
        if not isinstance(model, str) or not model:
            raise ValueError(f"Missing model in Replicate record: {record_path}")
        if not isinstance(task, str) or task not in REPLICATE_TASKS:
            raise ValueError(
                f"Unexpected Replicate CompBench task in {record_path}: {task!r}"
            )
        if not isinstance(sample_id, str) or sample_id not in selection_by_id:
            raise ValueError(f"Unknown sample_id in Replicate record: {record_path}")
        sample = selection_by_id[sample_id]
        if (
            sample.get("task") != task
            or sample.get("image_path") != record.get("image_path")
            or sample.get("instruction") != record.get("prompt")
        ):
            raise ValueError(f"Replicate record/selection mismatch: {record_path}")
        unique_key = (model, sample_id)
        if unique_key in seen:
            raise ValueError(f"Duplicate Replicate result for {model}/{sample_id}")
        seen.add(unique_key)
        outputs.append(
            {
                "model": model,
                "task": task,
                "sample_id": sample_id,
                "instruction": record["prompt"],
                "image_path": record["image_path"],
                "output_path": _resolve_under_root(
                    root, record.get("output_path"), "output_path"
                ),
                "source_path": _resolve_under_root(
                    root, sample.get("input_image"), "input_image"
                ),
                "mask_path": _resolve_under_root(root, sample.get("mask"), "mask"),
            }
        )
    if not outputs:
        raise FileNotFoundError(
            f"No succeeded Replicate records found under {records_root}"
        )
    return outputs


def evaluate_replicate_compbench_outputs(
    replicate_root: Path,
    *,
    scorer: Any | None = None,
    viescore_root: Path | None = None,
    key_path: Path | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Score successful Replicate CompBench multi-object output records."""
    root = replicate_root.expanduser().resolve()
    saved_outputs = collect_replicate_saved_outputs(root)
    scorer = _resolve_viescore(scorer, viescore_root, key_path)
    metric_path = root / "metrics" / "modified_viescore.jsonl"
    cached = _load_cached_scores(metric_path)
    rows: list[dict[str, Any]] = []
    for saved in tqdm(
        saved_outputs, desc="Replicate CompBench VIEScore", unit="sample"
    ):
        cache_key = _cache_key(
            saved,
            output_path=saved["output_path"],
            instruction=saved["instruction"],
            prompt_mode=None,
            model=saved["model"],
        )
        if cache_key in cached:
            score_row = cached[cache_key]
        else:
            with Image.open(saved["source_path"]) as source_file:
                source = source_file.copy()
            with Image.open(saved["mask_path"]) as mask_file:
                mask = mask_file.convert("L")
            scores, roi_box = _score_crop_pair(
                scorer,
                source,
                saved["output_path"],
                mask,
                saved["instruction"],
            )
            output_stat = saved["output_path"].stat()
            score_row = {
                "sample_id": saved["sample_id"],
                "model": saved["model"],
                "task": saved["task"],
                "instruction": saved["instruction"],
                "image_path": saved["image_path"],
                "output_path": str(saved["output_path"]),
                "output_size_bytes": output_stat.st_size,
                "output_mtime_ns": output_stat.st_mtime_ns,
                "roi_bbox_xyxy": list(roi_box),
                "sc": scores[0],
                "pq": scores[1],
                "o": scores[2],
            }
        rows.append(score_row)
        _write_jsonl(metric_path, rows)

    _write_jsonl(metric_path, rows)
    summary = _summarize(rows)
    summary_path = root / "metrics" / "modified_viescore_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Scored {len(rows)} succeeded Replicate CompBench multi-object outputs.")
    print(f"Per-sample VIEScore: {metric_path}")
    print(f"Model/task summary: {summary_path}")
    return rows, summary
