"""Submit first-turn generation as a Gemini Batch job, or collect a finished one.

Examples:
  uv run python src/scripts/generate_first_turn_batch.py --benchmark refedit_bench \
      --first-turn-file resources/refedit_bench_mixed_first_turn.jsonl --dry-run
  uv run python src/scripts/generate_first_turn_batch.py --benchmark refedit_bench \
      --first-turn-file resources/refedit_bench_mixed_first_turn.jsonl
  uv run python src/scripts/generate_first_turn_batch.py --benchmark compbench --prompt-mode text \
      --first-turn-file <file> --samples-file failed_samples.txt --dry-run
  uv run python src/scripts/generate_first_turn_batch.py --collect <work_dir>
  uv run python src/scripts/generate_first_turn_batch.py --collect <work_dir> --output-file <batch_output.jsonl>
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--collect", type=Path, help="Work directory of a submitted batch to collect.")
    parser.add_argument("--benchmark", choices=["refedit_bench", "compbench", "mira_bench"])
    parser.add_argument("--prompt-mode", default="mixed", choices=["mixed", "text"])
    parser.add_argument("--first-turn-file", type=Path, help="FirstTurnStore JSONL to read and write.")
    parser.add_argument("--model", help="Model id; defaults to MMEDIT_MODEL_ID.")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--samples-file", type=Path, help="Only generate the metadata.row_index values listed in this text file.")
    parser.add_argument("--work-dir", type=Path, help="Directory for batch inputs and state.")
    parser.add_argument("--dry-run", action="store_true", help="Write batch inputs without submitting.")
    parser.add_argument("--output-file", type=Path, help="Batch output to import (with --collect); defaults to <work_dir>/batch_output.jsonl.")
    parser.add_argument("--store-path", type=Path, help="Override the manifest store path (with --collect).")
    args = parser.parse_args()
    if not args.collect and (not args.benchmark or not args.first_turn_file):
        parser.error("--benchmark and --first-turn-file are required unless --collect is given.")
    if not args.collect and (args.output_file or args.store_path):
        parser.error("--output-file and --store-path require --collect.")
    if args.samples_file and (args.start_index != 0 or args.limit is not None):
        parser.error("--samples-file cannot be combined with --start-index or --limit.")
    return args


def _collect(work_dir: Path, output_file: Path | None, store_path: Path | None) -> None:
    from evaluation.first_turn_batch import collect_results, load_manifest
    from evaluation.notebook import create_agent_model
    from framework.core import Agent

    manifest = load_manifest(work_dir)
    parser = Agent(model=create_agent_model(manifest["model_id"]), verbosity_level=0)
    report = collect_results(work_dir, parser=parser, output_file=output_file, store_path=store_path)
    print(f"Store: {store_path or manifest['store_path']}")
    print(f"Stored: {report.stored}")
    print(f"Failed: {len(report.failed)}")
    for custom_id, reason in report.failed:
        print(f"  {custom_id}: {reason}")
    print(f"Missing from output: {len(report.missing)}")


def main() -> None:
    args = _parse_args()

    from framework.utils.project import load_project_environment

    load_project_environment()

    if args.collect:
        _collect(args.collect, args.output_file, args.store_path)
        return

    from evaluation.first_turn import FirstTurnStore
    from evaluation.compbench_incomplete import read_sample_indices
    from evaluation.first_turn_batch import (
        build_requests,
        submit_batch,
        write_batch_inputs,
    )
    from evaluation.notebook import create_agent_model, load_benchmark
    from framework.tools import image_editing_tools

    context = load_benchmark(args.benchmark, prompt_mode=args.prompt_mode)
    if args.samples_file:
        sample_indices = read_sample_indices(args.samples_file.expanduser().resolve(), context.records)
    else:
        total = len(context.records)
        stop = total if args.limit is None else min(total, args.start_index + args.limit)
        sample_indices = range(args.start_index, stop)
    model = create_agent_model(args.model)
    store = FirstTurnStore(args.first_turn_file)
    report = build_requests(
        context.records,
        sample_indices,
        benchmark=args.benchmark,
        prompt_mode=args.prompt_mode,
        image_loader=context.image_loader,
        model=model,
        tools=image_editing_tools(prompt_mode=args.prompt_mode),
        store=store,
    )
    print(f"Samples selected: {len(sample_indices)}")
    print(f"Already cached (skipped): {report.cached}")
    print(f"Requests to generate: {len(report.requests)}")
    for custom_id, reason in report.errors:
        print(f"  prepare error {custom_id}: {reason}")
    if not report.requests:
        print("Nothing to submit.")
        return

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    work_dir = args.work_dir or (
        Path("annotations") / "first_turn_batches" / f"{args.benchmark}_{args.prompt_mode}_{stamp}"
    )
    write_batch_inputs(
        work_dir,
        report.requests,
        benchmark=args.benchmark,
        model_id=model.model_id,
        store_path=args.first_turn_file,
    )
    print(f"Batch inputs written to {work_dir}")
    if args.dry_run:
        print("Dry run: batch not submitted.")
        return

    returncode = submit_batch(work_dir)
    if returncode != 0:
        raise SystemExit(f"submit_batch.py exited with code {returncode}. Check the output in {work_dir}.")
    print(f"Batch output written to {work_dir / 'batch_output.jsonl'}")
    print(f"Collect with: uv run python src/scripts/generate_first_turn_batch.py --collect {work_dir}")


if __name__ == "__main__":
    main()
