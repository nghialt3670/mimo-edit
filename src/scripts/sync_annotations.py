from __future__ import annotations

import argparse
import json
import os
import stat
import tempfile
from pathlib import Path
from typing import Any

if __package__:
    from .annotation_targets import ROOT, SUPPORTED_BENCHMARKS, annotation_target
else:
    from annotation_targets import ROOT, SUPPORTED_BENCHMARKS, annotation_target


def _read_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSONL in {path} at line {line_number}"
                ) from exc
            metadata = record.get("metadata")
            row_index = (
                metadata.get("row_index") if isinstance(metadata, dict) else None
            )
            if not isinstance(row_index, int) or isinstance(row_index, bool):
                raise ValueError(
                    f"Missing or invalid metadata.row_index in {path} at line {line_number}"
                )
            records.append(record)
    return records


def _read_successful_ids(path: Path) -> set[str]:
    successful_ids: set[str] = set()
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSONL in {path} at line {line_number}"
                ) from exc
            sample_id = record.get("sample_id")
            if sample_id is not None:
                successful_ids.add(str(sample_id))
    return successful_ids


def _atomic_write(path: Path, lines: list[str]) -> None:
    original_mode = stat.S_IMODE(path.stat().st_mode)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            temporary_file.writelines(lines)
        os.chmod(temporary_path, original_mode)
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def sync_annotations(
    annotations_path: Path,
    audit_path: Path,
    *,
    apply: bool = False,
) -> dict[str, int | bool]:
    """Remove resolved audit rows and sort successful annotations by source row."""
    records = _read_records(annotations_path)
    sorted_records = sorted(records, key=lambda record: record["metadata"]["row_index"])
    sort_changed = records != sorted_records

    successful_ids = {
        str(record["sample_id"])
        for record in sorted_records
        if record.get("sample_id") is not None
    }
    retained_audit_lines: list[str] = []
    removed_audit_count = 0
    with audit_path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                retained_audit_lines.append(line)
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSONL in {audit_path} at line {line_number}"
                ) from exc
            sample_id = record.get("sample_id")
            if sample_id is not None and str(sample_id) in successful_ids:
                removed_audit_count += 1
            else:
                retained_audit_lines.append(line)

    if apply:
        if sort_changed:
            _atomic_write(
                annotations_path,
                [
                    json.dumps(record, ensure_ascii=False) + "\n"
                    for record in sorted_records
                ],
            )
        if removed_audit_count:
            _atomic_write(audit_path, retained_audit_lines)

    return {
        "annotations": len(records),
        "sorted": sort_changed,
        "audit_removed": removed_audit_count,
        "audit_retained": len(retained_audit_lines),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Sync a benchmark's annotation audit against successful rows, then "
            "sort annotations by metadata.row_index."
        )
    )
    parser.add_argument("benchmark", choices=SUPPORTED_BENCHMARKS)
    parser.add_argument(
        "--annotations",
        type=Path,
        help="Successful annotation JSONL (defaults to the selected benchmark's output).",
    )
    parser.add_argument(
        "--audit",
        type=Path,
        help="Audit JSONL (defaults to the selected benchmark's audit file).",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Write changes; without this option, only report changes that would be made.",
    )
    args = parser.parse_args()

    target = annotation_target(args.benchmark, ROOT)
    annotations_path = (args.annotations or target.annotations).expanduser().resolve()
    audit_path = (args.audit or target.audit).expanduser().resolve()
    result = sync_annotations(annotations_path, audit_path, apply=args.apply)

    action = "Updated" if args.apply else "Would update"
    print(f"{action} annotations: {annotations_path}")
    print(
        f"{'Sorted' if args.apply else 'Would sort'} {result['annotations']} records: "
        f"{'changed' if result['sorted'] else 'already sorted'}."
    )
    print(
        f"{'Removed' if args.apply else 'Would remove'} {result['audit_removed']} "
        f"resolved audit entries; {result['audit_retained']} remain: {audit_path}"
    )
    if not args.apply and (result["sorted"] or result["audit_removed"]):
        print("Rerun with --apply to write the changes.")


if __name__ == "__main__":
    main()
