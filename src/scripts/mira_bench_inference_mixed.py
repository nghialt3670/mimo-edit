"""Run MIRA-Bench mixed-prompt inference from the command line."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

DATASET_ID = "ziqiangoodgood/MIRAGE"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run MixedPrompt mixed-prompt inference on MIRA-Bench annotations."
    )
    annotation_group = parser.add_mutually_exclusive_group()
    annotation_group.add_argument(
        "--annotations",
        type=Path,
        help="Annotation JSONL path; defaults to the latest MIRA-Bench annotation run.",
    )
    annotation_group.add_argument(
        "--annotation-run",
        help="MIRA-Bench annotation run name (run_<timestamp>) or 'latest'.",
    )
    parser.add_argument(
        "--image-root",
        type=Path,
        help=(
            "Directory with the MIRA-Bench benchmark images. Defaults to "
            "$MIRA_BENCH_IMAGE_ROOT, otherwise the images are downloaded from "
            f"the {DATASET_ID} Hugging Face dataset."
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        help=(
            "Output root; the 'mixed' subdirectory is appended. "
            "Defaults to .out/evaluation/mira_bench."
        ),
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="First annotation index to process (default: 0).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Process at most this many annotations, starting at --start-index.",
    )
    parser.add_argument(
        "--indices",
        type=int,
        nargs="+",
        metavar="INDEX",
        help="Run only these annotation indices (overrides --start-index/--limit).",
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
    args = parser.parse_args()

    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be greater than zero.")
    if args.start_index < 0:
        parser.error("--start-index must be zero or greater.")
    if args.indices and (args.start_index != 0 or args.limit is not None):
        parser.error("--indices cannot be combined with --start-index or --limit.")
    if args.indices and min(args.indices) < 0:
        parser.error("--indices must be zero or greater.")
    if args.run and not args.resume:
        parser.error("--run can only be used with --resume.")
    if args.rerun_completed and not args.resume:
        parser.error("--rerun-completed can only be used with --resume.")
    return args


def _resolve_launch_path(path: Path) -> Path:
    launch_dir = Path(os.environ.get("EVAL_START_DIR") or Path.cwd()).resolve()
    path = path.expanduser()
    return path.resolve() if path.is_absolute() else (launch_dir / path).resolve()


def _ensure_image_root(image_root: Path | None) -> None:
    """Point MIRA_BENCH_IMAGE_ROOT at local images, downloading them if needed."""
    if image_root is not None:
        os.environ["MIRA_BENCH_IMAGE_ROOT"] = str(_resolve_launch_path(image_root))
        return
    if os.environ.get("MIRA_BENCH_IMAGE_ROOT"):
        return

    from huggingface_hub import snapshot_download

    snapshot = snapshot_download(
        repo_id=DATASET_ID,
        repo_type="dataset",
        allow_patterns=[
            "benchmark/*.jpg",
            "benchmark/*.jpeg",
            "benchmark/*.png",
            "benchmark/*.webp",
        ],
    )
    os.environ["MIRA_BENCH_IMAGE_ROOT"] = os.path.join(snapshot, "benchmark")


def main() -> None:
    args = _parse_args()

    from framework.utils.project import load_project_environment

    load_project_environment()

    if args.annotations:
        os.environ["MIXED_PROMPT_MIRA_BENCH_ANNOTATIONS"] = str(
            _resolve_launch_path(args.annotations)
        )
    if args.annotation_run:
        os.environ["MIXED_PROMPT_MIRA_BENCH_ANNOTATIONS"] = ""
        os.environ["MIXED_PROMPT_MIRA_BENCH_ANNOTATION_RUN"] = args.annotation_run
    if args.output_root:
        os.environ["MIXED_PROMPT_MIRA_BENCH_OUTPUT_DIR"] = str(
            _resolve_launch_path(args.output_root)
        )
    _ensure_image_root(args.image_root)

    from evaluation.common import run_benchmark
    from evaluation.first_turn import build_first_turn_config, build_from_run
    from evaluation.notebook import (
        create_agent_model,
        create_run_directory,
        load_benchmark,
        resolve_run_directory,
    )
    from framework.tools import image_editing_tools

    context = load_benchmark("mira_bench", prompt_mode="mixed")
    total = len(context.records)
    if args.indices:
        out_of_range = [index for index in args.indices if index >= total]
        if out_of_range:
            raise ValueError(
                f"Indices {out_of_range} are out of range for {total} annotations."
            )
        sample_indices = list(dict.fromkeys(args.indices))
    else:
        stop = total if args.limit is None else min(total, args.start_index + args.limit)
        sample_indices = list(range(args.start_index, stop))
    if not sample_indices:
        raise ValueError("No MIRA-Bench annotations match the selected range.")

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

    print(f"Loaded {total} MIRA-Bench annotations from {context.annotation_file}")
    print(
        f"Processing {len(sample_indices)} selected annotations "
        f"(first index {sample_indices[0]}, last index {sample_indices[-1]})."
    )
    print(f"Evaluation run: {run_dir}")

    model = create_agent_model()
    tools = image_editing_tools()
    print(f"MIRA-Bench agent ready: {model.model_id}")

    first_turn = build_first_turn_config(args, run_dir)
    from_run = build_from_run(args)
    results = run_benchmark(
        context.records,
        sample_indices,
        benchmark="mira_bench",
        prompt_mode="mixed",
        output_dir=run_dir,
        image_loader=context.image_loader,
        model=model,
        tools=tools,
        resume=args.resume,
        rerun_completed=args.rerun_completed,
        display_fn=None,
        first_turn=first_turn,
        from_run=from_run,
    )

    completed = sum(result["status"] == "completed" for result in results)
    failed = sum(result["status"] == "failed" for result in results)
    print(
        f"MIRA-Bench samples: {len(results)} | "
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
