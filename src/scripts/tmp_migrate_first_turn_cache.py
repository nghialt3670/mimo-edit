"""Temporary: add prompt_signature to first-turn cache entries. No model calls."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from evaluation.first_turn import prompt_signature_from_task


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        help="Output JSONL (defaults to <input stem>.migrated.jsonl next to input).",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    input_path = args.input.expanduser().resolve()
    output_path = (
        args.output.expanduser().resolve()
        if args.output
        else input_path.with_name(f"{input_path.stem}.migrated.jsonl")
    )

    lines_out: list[str] = []
    stats: Counter[str] = Counter()
    for line_number, line in enumerate(input_path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON at {input_path}:{line_number}") from exc
        if "prompt_signature" in record:
            stats["already_migrated"] += 1
        else:
            record["prompt_signature"] = prompt_signature_from_task(record.get("task", ""))
            stats["migrated"] += 1
            if not record["prompt_signature"]:
                stats["empty_signature"] += 1
        lines_out.append(json.dumps(record, ensure_ascii=False))

    print(dict(stats), f"entries={len(lines_out)}")
    if args.dry_run:
        print("dry run: nothing written")
        return
    output_path.write_text("\n".join(lines_out) + "\n", encoding="utf-8")
    print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
