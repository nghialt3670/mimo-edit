from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

from PIL import Image
from tqdm.auto import tqdm

from evaluation.common import HFImageLoader, read_jsonl
from evaluation.notebook import resolve_run_directory
from evaluation.refedit_bench_metrics import (
    crop_pair_to_mask_roi,
    summarize_viescore_scores,
)


def _safe_sample_id(sample_id: Any) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(sample_id)).strip("._") or "sample"


def collect_saved_outputs(
    records: list[dict[str, Any]],
    results_path: Path,
    images_dir: Path,
) -> list[dict[str, Any]]:
    """Collect completed results and output images absent from a replaced summary."""
    outputs: dict[int, dict[str, Any]] = {}
    if results_path.is_file():
        for result in read_jsonl(results_path):
            if result.get("status") != "completed":
                continue
            index = result.get("annotation_index")
            if not isinstance(index, int) or not 0 <= index < len(records):
                raise ValueError(f"Invalid annotation_index in {results_path}: {index!r}")
            if result.get("sample_id") != records[index].get("sample_id"):
                raise ValueError(
                    f"Result/annotation mismatch at index {index}: "
                    f"{result.get('sample_id')!r} != {records[index].get('sample_id')!r}"
                )
            output_path = Path(result.get("output_path", "")).expanduser()
            if output_path.is_file():
                outputs[index] = {
                    "annotation_index": index,
                    "sample_id": result["sample_id"],
                    "output_path": output_path,
                    "discovered_from": "results.jsonl",
                }

    for output_path in sorted(images_dir.glob("*_output.png")):
        prefix, separator, _ = output_path.name.partition("_")
        if not separator or not prefix.isdigit():
            continue
        index = int(prefix)
        if not 0 <= index < len(records):
            raise ValueError(
                f"Saved output filename refers to missing annotation index {index}: "
                f"{output_path}"
            )
        expected_name = f"{index:06d}_{_safe_sample_id(records[index].get('sample_id'))}_output.png"
        if output_path.name != expected_name:
            raise ValueError(
                f"Saved output filename does not match annotation {index}: "
                f"{output_path.name!r} != {expected_name!r}"
            )
        outputs.setdefault(
            index,
            {
                "annotation_index": index,
                "sample_id": records[index].get("sample_id"),
                "output_path": output_path,
                "discovered_from": "output image",
            },
        )
    if not outputs:
        raise FileNotFoundError(
            f"No completed results or saved *_output.png images found in {images_dir}"
        )
    return [outputs[index] for index in sorted(outputs)]


def _load_viescore(viescore_root: Path, key_path: Path) -> Any:
    if not (viescore_root / "viescore" / "__init__.py").is_file():
        raise FileNotFoundError(
            f"VIESCORE_ROOT must contain viescore/__init__.py: {viescore_root}"
        )
    if not key_path.is_file():
        raise FileNotFoundError(
            f"VIESCORE_KEY_PATH must point to a file containing the OpenAI API key: {key_path}"
        )
    for import_path in (viescore_root, viescore_root / "viescore"):
        import_path_text = str(import_path.resolve())
        if import_path_text not in sys.path:
            sys.path.insert(0, import_path_text)
    try:
        from viescore import VIEScore
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Could not import VIEScore. Install its GEdit-Bench dependencies "
            "(including regex and requests) in this Python environment."
        ) from exc

    scorer = VIEScore(backbone="gpt4o", task="tie", key_path=str(key_path))
    if not hasattr(scorer.model, "model_name"):
        raise RuntimeError(
            "This VIEScore version does not expose its API model name; "
            "cannot guarantee GPT-4o evaluation."
        )
    scorer.model.model_name = "gpt-4o"
    return scorer


def evaluate_saved_refedit_bench_outputs(
    records: list[dict[str, Any]],
    *,
    output_dir: Path,
    image_loader: HFImageLoader | None = None,
    scorer: Any | None = None,
    viescore_root: Path | None = None,
    key_path: Path | None = None,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, float | int]]]:
    """Score one RefEdit run, including images omitted from results.jsonl."""
    output_dir = output_dir.expanduser().resolve()
    results_path = output_dir / "results" / "results.jsonl"
    images_dir = output_dir / "results" / "images"
    saved_outputs = collect_saved_outputs(records, results_path, images_dir)
    if image_loader is None:
        image_loader = HFImageLoader()
    if scorer is None:
        viescore_root = viescore_root or Path(os.environ.get("VIESCORE_ROOT", "")).expanduser()
        key_path = key_path or Path(os.environ.get("VIESCORE_KEY_PATH", "")).expanduser()
        if not str(viescore_root) or str(viescore_root) == ".":
            raise EnvironmentError("Set VIESCORE_ROOT to the Step1X-Edit GEdit-Bench directory.")
        if not str(key_path) or str(key_path) == ".":
            raise EnvironmentError("Set VIESCORE_KEY_PATH to the local OpenAI API key file.")
        scorer = _load_viescore(viescore_root, key_path)

    metric_path = output_dir / "metrics" / "modified_viescore.jsonl"
    cached_scores: dict[tuple[Any, ...], dict[str, Any]] = {}
    if metric_path.is_file():
        for cached in read_jsonl(metric_path):
            cache_key = (
                cached.get("sample_id"),
                cached.get("output_path"),
                cached.get("output_size_bytes"),
                cached.get("output_mtime_ns"),
                cached.get("instruction"),
            )
            cached_scores[cache_key] = cached

    metric_rows = []
    for saved in tqdm(
        saved_outputs,
        desc="Modified VIEScore",
        unit="sample",
        total=len(saved_outputs),
    ):
        annotation_index = saved["annotation_index"]
        record = records[annotation_index]
        metadata = record.get("metadata") or {}
        instruction = metadata.get("edit_instruction_single")
        if not isinstance(instruction, str) or not instruction.strip():
            raise ValueError(
                f"Missing metadata.edit_instruction_single for {record.get('sample_id')}"
            )
        category = str(metadata.get("category", "")).lower()
        if category not in {"easy", "hard"}:
            raise ValueError(
                f"Unexpected RefEdit category for {record.get('sample_id')}: {category!r}"
            )
        output_path: Path = saved["output_path"]
        output_stat = output_path.stat()
        cache_key = (
            record.get("sample_id"),
            str(output_path),
            output_stat.st_size,
            output_stat.st_mtime_ns,
            instruction,
        )
        if cache_key in cached_scores:
            score_row = cached_scores[cache_key]
        else:
            source_reference = image_loader.parse_reference(record.get("source_ref"))
            if source_reference is None:
                raise ValueError(f"Missing source_ref for {record.get('sample_id')}")
            dataset_id, hf_split, row_index, image_field = source_reference
            hf_row = image_loader.get_row(dataset_id, hf_split, row_index)
            if str(hf_row.get("category", "")).lower() != category:
                raise ValueError(
                    f"Dataset/annotation category mismatch for {record.get('sample_id')}"
                )
            if hf_row.get("edit_instruction_single") != instruction:
                raise ValueError(
                    "Dataset/annotation single-instruction mismatch for "
                    f"{record.get('sample_id')}"
                )
            source_image = image_loader.as_image(hf_row[image_field])
            with Image.open(output_path) as generated_image:
                source_roi, edited_roi, roi_box = crop_pair_to_mask_roi(
                    source_image, generated_image, hf_row["mask"]
                )
            scores = scorer.evaluate(
                [source_roi, edited_roi],
                instruction,
                extract_all_score=True,
            )
            if not isinstance(scores, (list, tuple)) or len(scores) != 3:
                raise ValueError(f"Expected VIEScore [SC, PQ, O], got {scores!r}")
            score_row = {
                "sample_id": record["sample_id"],
                "annotation_index": annotation_index,
                "category": category,
                "instruction": instruction,
                "output_path": str(output_path),
                "output_size_bytes": output_stat.st_size,
                "output_mtime_ns": output_stat.st_mtime_ns,
                "roi_bbox_xyxy": list(roi_box),
                "sc": float(scores[0]),
                "pq": float(scores[1]),
                "o": float(scores[2]),
            }
        summarize_viescore_scores([score_row])
        metric_rows.append(score_row)
        metric_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = metric_path.with_suffix(".jsonl.tmp")
        temporary_path.write_text(
            "".join(
                json.dumps(row, ensure_ascii=False) + "\n" for row in metric_rows
            ),
            encoding="utf-8",
        )
        temporary_path.replace(metric_path)

    summary = summarize_viescore_scores(metric_rows)
    print(
        f"Scored {len(metric_rows)} saved RefEdit outputs "
        f"({sum(row['category'] == 'easy' for row in metric_rows)} easy, "
        f"{sum(row['category'] == 'hard' for row in metric_rows)} hard)."
    )
    print(f"Per-sample Modified VIEScore: {metric_path}")
    return metric_rows, summary


def main() -> None:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parents[2] / ".env")
    parser = argparse.ArgumentParser(
        description="Run Modified VIEScore on existing RefEdit output images."
    )
    parser.add_argument(
        "--annotations",
        type=Path,
        default=Path(
            os.environ.get(
                "MMEDIT_REFEDIT_BENCH_ANNOTATIONS",
                ".out/annotation/refedit_bench/refedit_bench_mixed.jsonl",
            )
        ).expanduser(),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(".out/evaluation/refedit_bench"),
        help="Benchmark output root containing run_<timestamp> directories.",
    )
    parser.add_argument(
        "--run",
        default=os.environ.get("MMEDIT_REFEDIT_BENCH_RUN") or None,
        help="Run directory name to score; defaults to the latest completed run.",
    )
    parser.add_argument(
        "--viescore-root",
        type=Path,
        default=Path(os.environ.get("VIESCORE_ROOT", "")).expanduser(),
    )
    parser.add_argument(
        "--key-path",
        type=Path,
        default=Path(os.environ.get("VIESCORE_KEY_PATH", "")).expanduser(),
    )
    args = parser.parse_args()
    records = list(read_jsonl(args.annotations))
    if not records:
        raise ValueError(f"No RefEdit annotations found in {args.annotations}")
    _, summary = evaluate_saved_refedit_bench_outputs(
        records,
        output_dir=resolve_run_directory(args.output_dir, args.run),
        viescore_root=args.viescore_root,
        key_path=args.key_path,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
