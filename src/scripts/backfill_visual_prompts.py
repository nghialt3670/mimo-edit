from __future__ import annotations

import argparse
from pathlib import Path

from visualization import backfill_visual_prompts

from .annotation_targets import ROOT, SUPPORTED_BENCHMARKS, annotation_target


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Backfill box, point, and scribble prompts into one benchmark's "
            "annotation JSONL references."
        )
    )
    parser.add_argument(
        "benchmark",
        choices=SUPPORTED_BENCHMARKS,
        help="Benchmark whose annotations should receive visual prompts.",
    )
    parser.add_argument(
        "--annotations",
        type=Path,
        help="Annotation JSONL to update (defaults to the selected benchmark output).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate prompt geometry even when valid prompts are already present.",
    )
    args = parser.parse_args()

    target = annotation_target(args.benchmark, ROOT)
    annotations_path = (args.annotations or target.annotations).expanduser().resolve()
    result = backfill_visual_prompts(annotations_path, overwrite=args.overwrite)
    print(result)


if __name__ == "__main__":
    main()
