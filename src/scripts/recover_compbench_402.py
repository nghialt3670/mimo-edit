#!/usr/bin/env python3
"""Safely make confirmed Replicate HTTP 402 rejections retryable in CompBench v2.1.

It NEVER submits paid predictions. It changes only records with:
  status == 'create_uncertain'
  prediction_id is absent
  recorded submission error explicitly contains HTTP status 402 and insufficient credit

All modified record files get a .json.bak copy. Preview by default.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from collections import Counter
from pathlib import Path


def confirmed_402(error: object) -> bool:
    message = str(error or "")
    return bool(
        re.search(
            r"\b(?:status|status_code)\s*[:=]\s*402\b|\b402\s+(?:Payment Required|Insufficient credit)\b",
            message,
            re.I,
        )
        and re.search(r"insufficient\s+credit|payment\s+required", message, re.I)
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", default="compbench_replicate_results")
    p.add_argument(
        "--apply",
        action="store_true",
        help="Back up and mark only confirmed 402 records retryable",
    )
    a = p.parse_args()
    records = Path(a.output_dir).expanduser() / "records"
    if not records.is_dir():
        p.error(f"Record directory not found: {records}")

    counts = Counter()
    for path in sorted(records.rglob("*.json")):
        try:
            r = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            counts["unreadable_records"] += 1
            continue
        counts[str(r.get("status", "missing_status"))] += 1
        if (
            r.get("status") != "create_uncertain"
            or r.get("prediction_id")
            or not confirmed_402(r.get("error"))
        ):
            continue
        counts["confirmed_402"] += 1
        if a.apply:
            backup = path.with_suffix(path.suffix + ".bak")
            if not backup.exists():
                shutil.copy2(path, backup)
            r["status"] = "credit_exhausted"  # v2.1 will retry absent-id records
            tmp = path.with_suffix(path.suffix + ".tmp")
            tmp.write_text(
                json.dumps(r, ensure_ascii=False, indent=2, default=str) + "\n",
                encoding="utf-8",
            )
            tmp.replace(path)
            counts["updated"] += 1

    print("Record statuses (before repair):")
    for k, n in sorted(counts.items()):
        print(f"  {k}: {n}")
    if a.apply:
        print(
            f"Marked {counts['updated']} confirmed HTTP 402 records retryable; original .json.bak files retained."
        )
    else:
        print(
            "Preview only; no files changed. Use --apply to repair confirmed HTTP 402 records."
        )
    print("Other create_uncertain records are NOT touched; do not blindly retry them.")


if __name__ == "__main__":
    main()
