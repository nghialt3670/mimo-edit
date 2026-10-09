from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Iterable

from tqdm.auto import tqdm

from .core import Annotator, NoReferenceError
from .io import append_jsonl, save_reference_assets
from .schema import Sample

Adapter = Callable[[dict, str, int], Sample]


def _load_resume_state(output: Path, failures: Path) -> tuple[set[str], dict[str, str]]:
    accepted_ids: set[str] = set()
    previous_status: dict[str, str] = {}

    for path, is_output in ((output, True), (failures, False)):
        if not path.exists():
            continue
        with path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSONL in {path} at line {line_number}") from exc
                sample_id = record.get("sample_id")
                if sample_id is None:
                    continue
                sample_id = str(sample_id)
                if is_output:
                    accepted_ids.add(sample_id)
                else:
                    previous_status[sample_id] = str(record.get("status", "skipped"))

    return accepted_ids, previous_status


def convert_rows(
    rows: Iterable[dict],
    adapter: Adapter,
    split: str,
    annotator: Annotator,
    output: str | Path,
    failures: str | Path | None = None,
    limit: int | None = None,
    save_assets: bool = True,
) -> dict[str, int]:
    """Append new results, skipping completed samples and retrying audited samples.

    ``limit`` caps attempted samples, not rows already present in the output or
    audit files, so repeated runs continue through the dataset.
    """
    output = Path(output)
    failures = Path(failures) if failures else output.with_name(output.stem + "_audit.jsonl")
    output.parent.mkdir(parents=True, exist_ok=True)
    accepted_ids, previous_status = _load_resume_state(output, failures)

    counts = {"accepted": 0, "skipped": 0, "failed": 0, "resumed": 0, "retried": 0}
    iterator = enumerate(rows)
    attempted_count = 0
    attempted_ids: set[str] = set()

    for i, row in tqdm(iterator, desc=split):
        sample = None
        try:
            sample = adapter(row, split, i)
            sample_id = str(sample.sample_id)
            status = previous_status.get(sample_id)
            if sample_id in accepted_ids or sample_id in attempted_ids:
                counts["resumed"] += 1
                continue
            if limit is not None and attempted_count >= limit:
                break
            is_retry = status in {"failed", "skipped"}
            if is_retry:
                counts["retried"] += 1
            attempted_count += 1
            attempted_ids.add(sample_id)

            record = annotator.annotate(sample)
            if save_assets:
                save_reference_assets(sample, record, output)
            append_jsonl(output, record.model_dump(mode="json"))
            counts["accepted"] += 1
            accepted_ids.add(sample_id)
        except NoReferenceError as exc:
            counts["skipped"] += 1
            append_jsonl(
                failures,
                {
                    "status": "skipped",
                    "row_index": i,
                    "sample_id": getattr(sample, "sample_id", None),
                    "instruction": getattr(sample, "instruction", row.get("editing_instruction") or row.get("instruction")),
                    "error": str(exc),
                },
            )
            if sample is not None:
                previous_status[str(sample.sample_id)] = "skipped"
        except Exception as exc:
            counts["failed"] += 1
            append_jsonl(
                failures,
                {
                    "status": "failed",
                    "row_index": i,
                    "sample_id": getattr(sample, "sample_id", None),
                    "instruction": getattr(sample, "instruction", row.get("editing_instruction") or row.get("instruction")),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
            )
            if sample is not None:
                previous_status[str(sample.sample_id)] = "failed"
    return counts
