from __future__ import annotations

from collections.abc import Iterable
import re
from pathlib import Path
from typing import Any


def find_incomplete_compbench_samples(
    records: list[dict[str, Any]],
    results: Iterable[dict[str, Any]],
    run_dir: Path,
) -> list[tuple[int, str]]:
    """Return annotation positions lacking a completed output artifact."""
    positions_by_sample_id: dict[str, int] = {}
    for position, record in enumerate(records):
        sample_id = record.get("sample_id")
        if isinstance(sample_id, str):
            if sample_id in positions_by_sample_id:
                raise ValueError(f"Duplicate sample_id in annotations: {sample_id!r}")
            positions_by_sample_id[sample_id] = position

    results_by_index: dict[int, dict[str, Any]] = {}
    for result in results:
        sample_id = result.get("sample_id")
        if not isinstance(sample_id, str) or sample_id not in positions_by_sample_id:
            raise ValueError(
                f"Result has no matching annotation sample_id: {sample_id!r}"
            )
        position = positions_by_sample_id[sample_id]
        if position in results_by_index:
            raise ValueError(f"Duplicate result for sample_id {sample_id!r}")
        results_by_index[position] = result

    incomplete: list[tuple[int, str]] = []
    for index, record in enumerate(records):
        result = results_by_index.get(index)
        if result is None:
            reason = "missing_result"
        elif result.get("status") != "completed":
            reason = f"status_{result.get('status') or 'unknown'}"
        else:
            output_path_value = result.get("output_path")
            if not isinstance(output_path_value, str) or not output_path_value:
                reason = "missing_output_path"
            else:
                output_path = Path(output_path_value).expanduser()
                if not output_path.is_absolute():
                    output_path = run_dir / output_path
                if output_path.is_file():
                    reason = ""
                else:
                    safe_id = (
                        re.sub(
                            r"[^A-Za-z0-9._-]+",
                            "_",
                            str(record.get("sample_id") or f"annotation_{index}"),
                        ).strip("._")
                        or "sample"
                    )
                    renamed_outputs = list(
                        (run_dir / "results" / "images").glob(
                            f"*_{safe_id}_output.png"
                        )
                    )
                    reason = "" if len(renamed_outputs) == 1 else "missing_output_file"
        if reason:
            incomplete.append((index, reason))
    return incomplete


def read_sample_indices(path: Path, records: list[dict[str, Any]]) -> list[int]:
    """Read unique CompBench dataset row indices and return record positions."""
    if not path.is_file():
        raise FileNotFoundError(f"Sample index file not found: {path}")

    positions_by_row_index: dict[int, int] = {}
    for position, record in enumerate(records):
        metadata = record.get("metadata")
        row_index = (
            metadata.get("row_index") if isinstance(metadata, dict) else None
        )
        if not isinstance(row_index, int) or isinstance(row_index, bool):
            row_index = position
        if row_index in positions_by_row_index:
            raise ValueError(
                f"Duplicate CompBench metadata.row_index in annotations: "
                f"{row_index}"
            )
        positions_by_row_index[row_index] = position

    indices: list[int] = []
    seen: set[int] = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        value = line.partition("#")[0].strip()
        if not value:
            continue
        try:
            index = int(value)
        except ValueError as exc:
            raise ValueError(
                f"Expected a zero-based annotation index at {path}:{line_number}, "
                f"found {value!r}."
            ) from exc
        if index not in positions_by_row_index:
            raise ValueError(
                f"CompBench row index {index} at {path}:{line_number} is not "
                f"present in the annotation file."
            )
        if index in seen:
            raise ValueError(f"Duplicate CompBench row index {index} in {path}.")
        seen.add(index)
        indices.append(positions_by_row_index[index])
    if not indices:
        raise ValueError(f"No CompBench row indices found in {path}.")
    return indices


def write_sample_indices(
    path: Path,
    incomplete: list[tuple[int, str]],
    records: list[dict[str, Any]],
) -> None:
    """Write CompBench dataset row indices accepted by the mixed inference CLI."""
    lines = ["# CompBench metadata.row_index values; text after # is informational."]
    for index, reason in incomplete:
        sample_id = str(records[index].get("sample_id", f"annotation_{index}"))
        safe_sample_id = sample_id.replace("\n", " ").replace("\r", " ")
        record = records[index]
        metadata = record.get("metadata")
        row_index = (
            metadata.get("row_index") if isinstance(metadata, dict) else None
        )
        if not isinstance(row_index, int) or isinstance(row_index, bool):
            row_index = index
        lines.append(f"{row_index} # {reason}; sample_id={safe_sample_id}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
