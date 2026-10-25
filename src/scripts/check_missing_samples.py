"""List incomplete results in an inference run's results/ folder. Read-only; no API calls."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


def _safe_sample_id(sample_id: Any) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(sample_id)).strip("._") or "sample"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if line.strip():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path}:{line_no}: invalid JSON ({exc})") from exc
    return rows


def check_run(run_dir: Path, annotations_path: Path | None = None) -> dict[str, Any]:
    results_path = run_dir / "results" / "results.jsonl"
    images_dir = run_dir / "results" / "images"

    latest_by_sample: dict[str, dict[str, Any]] = {}
    if results_path.is_file():
        for row in _read_jsonl(results_path):
            sample_id = row.get("sample_id")
            if isinstance(sample_id, str):
                latest_by_sample[sample_id] = row  # last row wins, matching resume behaviour

    missing: list[dict[str, Any]] = []
    status_counts: Counter[str] = Counter()
    expected_outputs: set[str] = set()
    for sample_id, row in latest_by_sample.items():
        position = row.get("annotation_index")
        status = row.get("status") or "unknown"
        status_counts[status] += 1
        stem = f"{position:06d}_{_safe_sample_id(sample_id)}" if isinstance(position, int) else None
        if stem is not None:
            expected_outputs.add(f"{stem}_output.png")
        if status != "completed":
            reason = f"status={status}"
            if row.get("error"):
                reason += f" error={str(row['error'])[:200]}"
            missing.append({"position": position, "sample_id": sample_id, "reason": reason})
        elif stem is None or not (images_dir / f"{stem}_output.png").is_file():
            missing.append({"position": position, "sample_id": sample_id, "reason": "output image missing"})

    expected_total = None
    if annotations_path is not None:
        records = _read_jsonl(annotations_path)
        expected_total = len(records)
        for position, record in enumerate(records):
            sample_id = record.get("sample_id") or f"annotation_{position}"
            if sample_id not in latest_by_sample:
                status_counts["no_result_row"] += 1
                missing.append({"position": position, "sample_id": sample_id, "reason": "no result row"})

    orphan_images = sorted(
        path.name
        for path in images_dir.glob("*_output.png")
        if path.name not in expected_outputs
    ) if images_dir.is_dir() else []
    for name in orphan_images:
        missing.append({"position": None, "sample_id": None, "reason": f"image without result row: {name}"})

    return {
        "run_dir": str(run_dir),
        "expected_total": expected_total,
        "result_rows": len(latest_by_sample),
        "status_counts": dict(status_counts),
        "missing": missing,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, help="Inference run directory containing results/.")
    parser.add_argument("--annotations", type=Path, help="Annotation JSONL to compare against; enables no-row detection.")
    parser.add_argument("--write", type=Path, help="Optional path to write the missing list as JSONL.")
    args = parser.parse_args()

    run_dir = args.run_dir.expanduser().resolve()
    if not run_dir.is_dir():
        raise SystemExit(f"run directory not found: {run_dir}")
    annotations = args.annotations.expanduser().resolve() if args.annotations else None
    if annotations is not None and not annotations.is_file():
        raise SystemExit(f"annotations not found: {annotations}")

    report = check_run(run_dir, annotations)
    print(f"run: {report['run_dir']}")
    if report["expected_total"] is not None:
        print(f"expected samples: {report['expected_total']}")
    print(f"result rows: {report['result_rows']}")
    print(f"status counts: {report['status_counts']}")
    print(f"missing: {len(report['missing'])}")
    for item in report["missing"]:
        label = item["sample_id"] or "-"
        print(f"  [{item['position']}] {label}: {item['reason']}")

    if args.write:
        args.write.parent.mkdir(parents=True, exist_ok=True)
        args.write.write_text(
            "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in report["missing"]),
            encoding="utf-8",
        )
        print(f"wrote: {args.write}")


if __name__ == "__main__":
    main()
