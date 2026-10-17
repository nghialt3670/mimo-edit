#!/usr/bin/env python3
"""Recover unfinished CompBench Replicate v2.1 jobs without touching successes.

Read-only by default. `--apply` repairs confirmed rejected creation requests
(HTTP 402/429) and optionally links a unique match from Replicate's prediction
history. `--force-ambiguous` additionally permits resubmitting jobs whose
original creation outcome cannot be verified; this CAN cause duplicate charges.

After this tool, run the original compbench_replicate.py with --retry-failed.
The original runner handles polling, image download, and file layout.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import urlparse

STATUS_RE = re.compile(
    r"\b(?:status|status_code)\s*[:=]\s*(402|429)\b|\b(402|429)\s+(?:Payment Required|Insufficient Credit|Too Many Requests)\b",
    re.I,
)
AMBIGUOUS_STATUSES = {"create_uncertain", "creating"}
RETRYABLE_STATUSES = {"rate_limited", "credit_exhausted", "retry_ambiguous"}
TERMINAL_FAILURES = {"failed", "canceled", "create_error"}


def status_code_from_error(record: dict) -> int | None:
    message = "\n".join(str(record.get(k) or "") for k in ("error", "create_error"))
    hit = STATUS_RE.search(message)
    if not hit:
        return None
    return int(hit.group(1) or hit.group(2))


def parse_date(iso: str | None) -> datetime | None:
    if not iso:
        return None
    try:
        dt = datetime.fromisoformat(iso.replace("Z", "+00:00"))
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except (TypeError, ValueError):
        return None


def list_recent_predictions(token: str, max_pages: int) -> list[dict]:
    """Read-only GET: no model inference or billing."""
    import requests

    url = "https://api.replicate.com/v1/predictions"
    records: list[dict] = []
    session = requests.Session()
    session.headers.update(
        {"Authorization": f"Bearer {token}", "Accept": "application/json"}
    )
    for _ in range(max_pages):
        if (
            urlparse(url).netloc != "api.replicate.com"
            or urlparse(url).path != "/v1/predictions"
        ):
            raise RuntimeError(f"Unexpected API pagination URL: {url}")
        r = session.get(url, timeout=(10, 30))
        r.raise_for_status()
        page = r.json()
        records.extend(page.get("results", []))
        url = page.get("next")
        if not url:
            break
    return records


def find_exact_match(record: dict, remote_predictions: list[dict]) -> list[dict]:
    """Conservative match: model, prompt, seed, and submission timestamp window."""
    submitted = parse_date(record.get("submitted_at"))
    if submitted is None:
        return []
    expected_slug = record.get("model_slug")
    expected_seed = record.get("seed")
    expected_prompt = record.get("prompt")
    if expected_slug is None or expected_seed is None or not expected_prompt:
        return []
    found = []
    for prediction in remote_predictions:
        if prediction.get("model") != expected_slug:
            continue
        inputs = prediction.get("input")
        if not isinstance(inputs, dict):
            continue  # Replicate removes prediction inputs after its retention window.
        if inputs.get("prompt") != expected_prompt:
            continue
        try:
            if int(inputs.get("seed")) != int(expected_seed):
                continue
        except (ValueError, TypeError):
            continue
        created = parse_date(prediction.get("created_at"))
        if created is None or not (
            submitted - timedelta(minutes=2)
            <= created
            <= submitted + timedelta(minutes=10)
        ):
            continue
        if prediction.get("id"):
            found.append(prediction)
    return found


def atomic_update(path: Path, row: dict) -> None:
    backup = path.with_suffix(path.suffix + ".bak")
    if not backup.exists():
        shutil.copy2(path, backup)
    target = path.with_name(path.name + ".recovery_tmp")
    target.write_text(
        json.dumps(row, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )
    target.replace(path)


def classify(record: dict) -> str:
    status = record.get("status", "missing_status")
    if status == "succeeded":
        return "complete"
    if record.get("prediction_id"):
        if status in TERMINAL_FAILURES:
            return "failed_prediction_retry_is_paid"
        return "has_prediction_id_resume_without_new_post"
    if status in AMBIGUOUS_STATUSES:
        code = status_code_from_error(record)
        if code in (402, 429):
            return f"definitely_rejected_http_{code}"
        return "AMBIGUOUS_no_prediction_id"
    if status in TERMINAL_FAILURES:
        return "failed_prediction_retry_is_paid"
    if status in RETRYABLE_STATUSES:
        return "already_retryable"
    return f"other_{status}"


def recover(
    root: Path,
    *,
    apply: bool,
    force_ambiguous: bool,
    remote_predictions: list[dict] | None = None,
    verbose: bool = True,
) -> dict:
    records_root = root / "records"
    if not records_root.is_dir():
        raise FileNotFoundError(f"No records directory: {records_root}")
    counts = Counter()
    actions = Counter()
    assigned_remote_ids = set()
    for path in sorted(records_root.rglob("*.json")):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            print(f"WARNING: unreadable record {path}: {exc}")
            counts["unreadable"] += 1
            continue
        category = classify(record)
        counts[category] += 1
        if category == "complete":
            continue
        # Keep the original result of every unfinished attempt for audit,
        # including `failed` records the benchmark runner will overwrite.
        if apply:
            backup = path.with_suffix(path.suffix + ".bak")
            if not backup.exists():
                shutil.copy2(path, backup)
                actions["backed_up"] += 1
        note = ""
        updated = dict(record)
        old_status = record.get("status")

        # A previously created prediction must be polled by the ORIGINAL runner;
        # don't submit a second POST here.
        if category.startswith("definitely_rejected_http_"):
            code = status_code_from_error(record)
            updated["status"] = "rate_limited" if code == 429 else "credit_exhausted"
            note = "SAFE: allow new prediction; original POST was rejected"
        elif category == "AMBIGUOUS_no_prediction_id":
            matches = (
                find_exact_match(record, remote_predictions)
                if remote_predictions is not None
                else []
            )
            if len(matches) == 1 and matches[0]["id"] not in assigned_remote_ids:
                remote = matches[0]
                updated["prediction_id"] = remote["id"]
                updated["status"] = "pending_retrieval"
                updated["resolved_version"] = remote.get("version")
                assigned_remote_ids.add(remote["id"])
                note = "SAFE: matched existing prediction; will fetch this ID"
            elif len(matches) > 1:
                note = (
                    f"UNRESOLVED: {len(matches)} remote matches; manual review needed"
                )
            elif force_ambiguous:
                updated["status"] = "retry_ambiguous"
                note = "RISK: resubmit without proof old POST was rejected; possible duplicate charge"
            else:
                note = "UNRESOLVED: can't verify whether a paid prediction was created"
        elif category == "failed_prediction_retry_is_paid":
            note = "Use --retry-failed with the original runner (may incur another inference charge)"
        elif category == "has_prediction_id_resume_without_new_post":
            note = "Original runner fetches this existing prediction ID"
        elif category == "already_retryable":
            note = "Original runner will submit this previously rejected job"
        else:
            note = "Inspect record before retry"

        if updated != record:
            actions["would_modify"] += 1
            if apply:
                atomic_update(path, updated)
                actions["modified"] += 1
        if verbose:
            print(
                f"{path.relative_to(records_root)}: {old_status}"
                f" | {category} | {note}"
                + (f" | -> {updated['status']}" if updated != record else "")
            )
    if verbose:
        print("\nClassification:")
        for category, count in sorted(counts.items()):
            print(f"  {category}: {count}")
        print(
            f"Would modify: {actions['would_modify']}; modified: {actions['modified']}"
        )
        if not apply:
            print(
                "PREVIEW ONLY: no files changed. Use --apply to safely repair confirmed 402/429."
            )
        else:
            print("Backups for modified records: *.json.bak")
    return {"counts": dict(counts), "actions": dict(actions)}


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--output-dir", default="compbench_replicate_results")
    p.add_argument(
        "--apply", action="store_true", help="Write changes; without this only preview"
    )
    p.add_argument(
        "--reconcile-replicate",
        action="store_true",
        help="Read-only Replicate history lookup for missing prediction IDs",
    )
    p.add_argument(
        "--max-pages",
        type=int,
        default=20,
        help="Maximum prediction-history pages (100 per page)",
    )
    p.add_argument(
        "--force-ambiguous",
        action="store_true",
        help="DANGEROUS: resubmit uncertain/no-ID jobs even when creation may have succeeded",
    )
    args = p.parse_args()
    if args.force_ambiguous and not args.apply:
        p.error("--force-ambiguous must be combined with --apply")
    if args.max_pages <= 0:
        p.error("--max-pages must be > 0")
    remote = None
    if args.reconcile_replicate:
        token = os.environ.get("REPLICATE_API_TOKEN")
        if not token:
            p.error("Set REPLICATE_API_TOKEN for --reconcile-replicate")
        remote = list_recent_predictions(token, args.max_pages)
        print(
            f"Loaded {len(remote)} Replicate prediction history entries via GET (no inference billing)."
        )
    recover(
        Path(args.output_dir).expanduser(),
        apply=args.apply,
        force_ambiguous=args.force_ambiguous,
        remote_predictions=remote,
    )


if __name__ == "__main__":
    main()
