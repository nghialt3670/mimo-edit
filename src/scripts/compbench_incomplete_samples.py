"""Write missing or incomplete CompBench inference indices to a text file."""

from __future__ import annotations

import argparse
import os
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "List CompBench annotations that have no completed inference result "
            "or whose output image is missing."
        )
    )
    annotation_group = parser.add_mutually_exclusive_group()
    annotation_group.add_argument("--annotations", type=Path)
    annotation_group.add_argument("--annotation-run")
    parser.add_argument(
        "--prompt-mode",
        choices=("text", "mixed"),
        default="mixed",
        help="Inference run to inspect (default: mixed).",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        help=(
            "CompBench evaluation output root; the selected prompt-mode directory "
            "is appended."
        ),
    )
    parser.add_argument(
        "--run",
        help="Inference run_<timestamp> to inspect; defaults to the latest run.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("compbench_incomplete_samples.txt"),
        help="Output text file (default: ./compbench_incomplete_samples.txt).",
    )
    return parser.parse_args()


def _resolve_launch_path(path: Path) -> Path:
    launch_dir = Path(os.environ.get("EVAL_START_DIR") or Path.cwd()).resolve()
    path = path.expanduser()
    return path.resolve() if path.is_absolute() else (launch_dir / path).resolve()


def main() -> None:
    args = _parse_args()

    from framework.utils.project import load_project_environment

    load_project_environment()

    if args.annotations:
        os.environ["MMEDIT_COMPBENCH_ANNOTATIONS"] = str(
            _resolve_launch_path(args.annotations)
        )
    if args.annotation_run:
        os.environ["MMEDIT_COMPBENCH_ANNOTATIONS"] = ""
        os.environ["MMEDIT_COMPBENCH_ANNOTATION_RUN"] = args.annotation_run
    if args.output_root:
        os.environ["MMEDIT_COMPBENCH_OUTPUT_DIR"] = str(
            _resolve_launch_path(args.output_root)
        )

    from evaluation.common import read_jsonl
    from evaluation.compbench_incomplete import (
        find_incomplete_compbench_samples,
        write_sample_indices,
    )
    from evaluation.notebook import load_benchmark, resolve_run_directory

    context = load_benchmark("compbench", prompt_mode=args.prompt_mode)
    run_dir = resolve_run_directory(context.output_dir, args.run)
    results_path = run_dir / "results" / "results.jsonl"
    incomplete = find_incomplete_compbench_samples(
        context.records,
        read_jsonl(results_path),
        run_dir,
    )
    output_path = _resolve_launch_path(args.output)
    write_sample_indices(output_path, incomplete, context.records)
    print(f"Prompt mode: {args.prompt_mode}")
    print(f"Inspected run: {run_dir}")
    print(f"Incomplete samples: {len(incomplete)} / {len(context.records)}")
    print(f"Sample list: {output_path}")


if __name__ == "__main__":
    main()
