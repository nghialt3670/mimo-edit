from __future__ import annotations

from collections.abc import Iterable
from typing import Any


def select_compbench_indices(
    records: list[dict[str, Any]],
    *,
    tasks: Iterable[str] | None = None,
    start_index: int = 0,
    limit: int | None = None,
) -> list[int]:
    """Select CompBench annotation indices by task and original JSONL position."""
    if start_index < 0:
        raise ValueError("start_index must be zero or greater")
    if limit is not None and limit < 1:
        raise ValueError("limit must be greater than zero")
    if records and start_index >= len(records):
        raise ValueError(
            f"start_index {start_index} is outside the annotation file, which has "
            f"{len(records)} records."
        )

    selected_tasks = set(tasks) if tasks is not None else None
    if selected_tasks is not None:
        if not selected_tasks or any(
            not isinstance(task, str) or not task for task in selected_tasks
        ):
            raise ValueError("tasks must contain one or more non-empty task names")
        available_tasks = {
            metadata["task"]
            for record in records
            if isinstance((metadata := record.get("metadata")), dict)
            and isinstance(metadata.get("task"), str)
        }
        unknown_tasks = selected_tasks - available_tasks
        if unknown_tasks:
            raise ValueError(
                f"Unknown CompBench task(s): {', '.join(sorted(unknown_tasks))}. "
                f"Available tasks: {', '.join(sorted(available_tasks))}"
            )

    indices = []
    for index, record in enumerate(records):
        if index < start_index:
            continue
        metadata = record.get("metadata")
        task = metadata.get("task") if isinstance(metadata, dict) else None
        if selected_tasks is None or task in selected_tasks:
            indices.append(index)
    return indices if limit is None else indices[:limit]
