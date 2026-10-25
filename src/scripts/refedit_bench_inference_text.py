"""Run RefEdit-Bench text-prompt inference from the command line."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from framework.inference import ModelPool


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run MMEdit text-prompt inference on RefEdit-Bench annotations."
    )
    annotation_group = parser.add_mutually_exclusive_group()
    annotation_group.add_argument(
        "--annotations",
        type=Path,
        help="Annotation JSONL path; defaults to the latest RefEdit-Bench annotation run.",
    )
    annotation_group.add_argument(
        "--annotation-run",
        help="RefEdit-Bench annotation run name (run_<timestamp>) or 'latest'.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        help=(
            "Output root; the 'text' subdirectory is appended. "
            "Defaults to .out/evaluation/refedit_bench."
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Process at most this many annotations, starting at --start-index.",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="First annotation index to process (default: 0).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume an existing evaluation run, skipping completed samples.",
    )
    parser.add_argument(
        "--run",
        help="Existing run_<timestamp> to resume; defaults to the latest run.",
    )
    parser.add_argument(
        "--rerun-completed",
        action="store_true",
        help=(
            "With --resume, rerun every selected sample after removing its old logs "
            "and generated images."
        ),
    )
    parser.add_argument(
        "--require-cuda",
        action="store_true",
        help="Fail before model loading if PyTorch cannot access a CUDA GPU.",
    )
    from evaluation.first_turn import add_first_turn_arguments

    add_first_turn_arguments(parser)
    args = parser.parse_args(argv)

    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be greater than zero.")
    if args.start_index < 0:
        parser.error("--start-index must be zero or greater.")
    if args.run and not args.resume:
        parser.error("--run can only be used with --resume.")
    if args.rerun_completed and not args.resume:
        parser.error("--rerun-completed can only be used with --resume.")
    if args.annotations and args.annotation_run:
        parser.error("--annotations and --annotation-run cannot be combined.")
    return args


def _resolve_launch_path(path: Path) -> Path:
    launch_dir = Path(os.environ.get("EVAL_START_DIR") or Path.cwd()).resolve()
    path = path.expanduser()
    return path.resolve() if path.is_absolute() else (launch_dir / path).resolve()


def main(argv: list[str] | None = None, pool: ModelPool | None = None) -> None:
    args = _parse_args(argv)

    from framework.utils.project import load_project_environment

    load_project_environment()

    if args.annotations:
        os.environ["MMEDIT_REFEDIT_BENCH_ANNOTATIONS"] = str(
            _resolve_launch_path(args.annotations)
        )
    if args.annotation_run:
        os.environ["MMEDIT_REFEDIT_BENCH_ANNOTATIONS"] = ""
        os.environ["MMEDIT_REFEDIT_BENCH_ANNOTATION_RUN"] = args.annotation_run
    if args.output_root:
        os.environ["MMEDIT_REFEDIT_BENCH_OUTPUT_DIR"] = str(
            _resolve_launch_path(args.output_root)
        )

    from evaluation.common import run_benchmark
    from evaluation.first_turn import build_first_turn_config
    from evaluation.notebook import (
        create_agent_model,
        create_run_directory,
        load_benchmark,
        resolve_run_directory,
    )
    from framework.tools import image_editing_tools

    context = load_benchmark("refedit_bench", prompt_mode="text")
    first_index = args.start_index
    stop_index = (
        len(context.records)
        if args.limit is None
        else min(len(context.records), first_index + args.limit)
    )
    if first_index >= len(context.records):
        raise ValueError(
            f"--start-index {first_index} is outside the annotation file, which has "
            f"{len(context.records)} records."
        )
    sample_indices = range(first_index, stop_index)

    if args.require_cuda:
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is required but PyTorch cannot access a CUDA device. "
                "Check the NVIDIA driver and installed PyTorch build."
            )
        print(f"CUDA GPU: {torch.cuda.get_device_name(0)}")

    if args.resume:
        run_dir = resolve_run_directory(context.output_dir, args.run)
    else:
        run_dir = create_run_directory(context.output_dir)

    print(
        f"Loaded {len(context.records)} RefEdit-Bench annotations from "
        f"{context.annotation_file}"
    )
    print(
        f"Processing annotation indices {first_index}..{stop_index - 1} "
        f"({stop_index - first_index} samples)"
    )
    print(f"Evaluation run: {run_dir}")

    model = create_agent_model()
    tools = image_editing_tools(pool=pool)
    print(f"RefEdit-Bench agent ready: {model.model_id}")

    first_turn = build_first_turn_config(args, run_dir)
    results = run_benchmark(
        context.records,
        sample_indices,
        benchmark="refedit_bench",
        prompt_mode="text",
        output_dir=run_dir,
        image_loader=context.image_loader,
        model=model,
        tools=tools,
        resume=args.resume,
        rerun_completed=args.rerun_completed,
        display_fn=None,
        first_turn=first_turn,
    )

    completed = sum(result["status"] == "completed" for result in results)
    failed = sum(result["status"] == "failed" for result in results)
    print(
        f"RefEdit-Bench samples: {len(results)} | "
        f"completed: {completed} | failed: {failed}"
    )
    if results:
        mean_runtime = sum(result["elapsed_seconds"] for result in results) / len(
            results
        )
        print(f"Mean runtime: {mean_runtime:.2f}s")
    print(f"Results: {run_dir / 'results' / 'results.jsonl'}")


if __name__ == "__main__":
    main()
