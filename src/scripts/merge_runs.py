"""Merge two inference runs into a new run.

Samples completed in --run-b override --run-a; samples only in --run-a are kept.
Results, output images, and cached VIEScore rows are copied into the new run.

Usage:
    uv run python src/scripts/merge_runs.py --run-a run_20261012_092541_755821 --run-b run_20261025_124646_118000
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

from evaluation.common import read_jsonl
from evaluation.notebook import create_run_directory, resolve_run_directory

DEFAULT_OUTPUT_ROOT = Path(".out/evaluation/refedit_bench/mixed")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-a", required=True, help="Base run, used for samples not completed in run-b.")
    parser.add_argument("--run-b", required=True, help="Priority run, overrides run-a on common samples.")
    return parser.parse_args()


def _completed_results(run_dir: Path) -> dict[str, dict[str, Any]]:
    results_path = run_dir / "results" / "results.jsonl"
    if not results_path.is_file():
        raise FileNotFoundError(f"Missing results: {results_path}")
    rows: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(results_path):
        if row.get("status") == "completed":
            rows[row["sample_id"]] = row
    return rows


def _cached_scores(run_dir: Path) -> dict[str, dict[str, Any]]:
    cache_path = run_dir / "metrics" / "modified_viescore.jsonl"
    if not cache_path.is_file():
        return {}
    return {row["sample_id"]: row for row in read_jsonl(cache_path)}


def _source_image(run_dir: Path, row: dict[str, Any]) -> Path | None:
    # Stored output_path prefix can be stale; on-disk prefix equals annotation_index.
    stored_name = Path(row.get("output_path") or "").name
    suffix = stored_name.split("_", 1)[1] if "_" in stored_name else stored_name
    candidate = run_dir / "results" / "images" / f"{row['annotation_index']:06d}_{suffix}"
    return candidate if candidate.is_file() else None


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    args = _parse_args()
    output_root = args.output_root.expanduser().resolve()
    run_a = resolve_run_directory(output_root, args.run_a)
    run_b = resolve_run_directory(output_root, args.run_b)
    if run_a == run_b:
        raise ValueError("--run-a and --run-b must be different runs")

    results = {run_a: _completed_results(run_a), run_b: _completed_results(run_b)}
    scores = {run_a: _cached_scores(run_a), run_b: _cached_scores(run_b)}

    # run-b is applied last, so its rows replace run-a rows for common samples.
    chosen: dict[str, tuple[Path, dict[str, Any]]] = {}
    for run_dir in (run_a, run_b):
        for sample_id, row in results[run_dir].items():
            chosen[sample_id] = (run_dir, row)

    merged_dir = create_run_directory(output_root)
    images_dir = merged_dir / "results" / "images"
    images_dir.mkdir(parents=True)
    (merged_dir / "metrics").mkdir()

    merged_results: list[dict[str, Any]] = []
    merged_scores: list[dict[str, Any]] = []
    from_counts = {run_a.name: 0, run_b.name: 0}
    skipped: list[str] = []

    for sample_id, (source_run, row) in chosen.items():
        image = _source_image(source_run, row)
        if image is None:
            skipped.append(sample_id)
            continue
        target = images_dir / image.name
        shutil.copy2(image, target)  # copy2 keeps mtime, so cached stat checks still match
        from_counts[source_run.name] += 1
        merged_results.append({**row, "output_path": str(target)})
        cached = scores[source_run].get(sample_id)
        if cached is not None:
            merged_scores.append({**cached, "output_path": str(target)})

    merged_results.sort(key=lambda row: row.get("annotation_index", 0))
    merged_scores.sort(key=lambda row: row.get("annotation_index", 0))
    _write_jsonl(merged_dir / "results" / "results.jsonl", merged_results)
    _write_jsonl(merged_dir / "metrics" / "modified_viescore.jsonl", merged_scores)

    print(f"Merged run: {merged_dir}")
    print(f"From {run_a.name}: {from_counts[run_a.name]}")
    print(f"From {run_b.name}: {from_counts[run_b.name]}")
    print(f"Cached VIEScore rows copied: {len(merged_scores)}")
    if skipped:
        print(f"Skipped (output image missing): {len(skipped)}")


if __name__ == "__main__":
    main()
