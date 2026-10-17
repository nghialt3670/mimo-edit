#!/usr/bin/env python3
"""Run MIRA-Bench image-editing inference through Replicate."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import time
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

from PIL import Image, ImageOps

from scripts.compbench_inference_replicate import (
    MODELS,
    SubmissionLimiter,
    create_with_rate_limit_retry,
    download_result_image,
    is_rejected_429,
    model_input,
    output_image_url,
    predict_until_done,
)

DATASET_ID = "ziqiangoodgood/MIRAGE"
ANNOTATION_FILE = "benchmark/annotations.jsonl"
DEFAULT_OUTPUT_ROOT = Path(".out/evaluation/mira_bench")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(path.name + ".tmp")
    temporary_path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(path)


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON record: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def safe_relative(image_name: str) -> Path:
    path = PurePosixPath(str(image_name).replace("\\", "/"))
    if (
        not image_name
        or path.is_absolute()
        or any(part in ("..", "", ".") for part in path.parts)
    ):
        raise ValueError(f"Unsafe MIRA-Bench image path: {image_name!r}")
    relative_path = Path(*path.parts)
    if relative_path.suffix.lower() not in IMAGE_SUFFIXES:
        raise ValueError(f"Unsupported MIRA-Bench image type: {image_name!r}")
    return relative_path


def sample_key(sample_id: str) -> str:
    return hashlib.sha256(sample_id.encode("utf-8")).hexdigest()[:20]


def seed_for(sample_id: str, base_seed: int) -> int:
    digest = hashlib.sha256(sample_id.encode("utf-8")).digest()
    return (base_seed + int.from_bytes(digest[:4], "big")) % (2**31 - 1)


def save_image(image: Image.Image, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    image = ImageOps.exif_transpose(image).convert("RGB")
    temporary_path = destination.with_name(destination.name + ".tmp")
    image.save(temporary_path, format="PNG")
    temporary_path.replace(destination)


def png_is_valid(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size == 0:
        return False
    try:
        with Image.open(path) as image:
            image.verify()
        return True
    except (OSError, ValueError):
        return False


def _read_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid MIRA-Bench JSONL at {path}:{line_number}"
                ) from exc
            if not isinstance(row, dict):
                raise ValueError(f"Expected a JSON object at {path}:{line_number}")
            rows.append(row)
    return rows


def load_dataset_rows(
    dataset_dir: Path | None = None,
) -> tuple[list[dict[str, Any]], Path | None]:
    """Load annotation rows and optional local image root."""
    if dataset_dir is not None:
        dataset_dir = dataset_dir.expanduser().resolve()
        benchmark_root = dataset_dir / "benchmark"
        if not benchmark_root.is_dir() and dataset_dir.name == "benchmark":
            benchmark_root = dataset_dir
        annotation_path = benchmark_root / "annotations.jsonl"
        if not annotation_path.is_file():
            raise FileNotFoundError(
                f"MIRA-Bench annotations not found under {dataset_dir}: "
                f"expected {annotation_path}"
            )
        return _read_rows(annotation_path), benchmark_root

    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise ImportError(
            "MIRA-Bench Replicate inference requires huggingface_hub. "
            "Install it in the project environment."
        ) from exc
    annotation_path = Path(
        hf_hub_download(
            repo_id=DATASET_ID,
            repo_type="dataset",
            filename=ANNOTATION_FILE,
        )
    )
    return _read_rows(annotation_path), None


def resolve_source_image(
    image_name: str,
    *,
    dataset_root: Path | None,
) -> Path:
    relative_path = safe_relative(image_name)
    if dataset_root is not None:
        source_path = (dataset_root / relative_path).resolve()
        if not source_path.is_relative_to(dataset_root.resolve()):
            raise ValueError(f"MIRA-Bench image escapes dataset root: {image_name!r}")
        if not source_path.is_file():
            raise FileNotFoundError(f"MIRA-Bench image not found: {source_path}")
        return source_path

    from huggingface_hub import hf_hub_download

    return Path(
        hf_hub_download(
            repo_id=DATASET_ID,
            repo_type="dataset",
            filename=f"benchmark/{relative_path.as_posix()}",
        )
    )


def prepare_samples(
    root: Path,
    rows: list[dict[str, Any]],
    *,
    dataset_root: Path | None = None,
    sample_ids: list[str] | None = None,
    sample_indices: list[int] | None = None,
    max_samples: int | None = None,
) -> list[dict[str, Any]]:
    """Resolve requested MIRA-Bench IDs and cache their source images locally."""
    if sample_ids and len(set(sample_ids)) != len(sample_ids):
        raise ValueError("sample_ids must not contain duplicates")
    if sample_indices and len(set(sample_indices)) != len(sample_indices):
        raise ValueError("sample_indices must not contain duplicates")
    if sample_ids and sample_indices:
        raise ValueError("Select samples by IDs or indices, not both")
    if sample_indices and any(index < 0 for index in sample_indices):
        raise ValueError("sample_indices must be non-negative")
    requested_ids = set(sample_ids or [])
    selected_rows_by_index: dict[int, tuple[str, str, str]] = {}
    seen_ids: set[str] = set()
    for row_index, row in enumerate(rows):
        if sample_indices and row_index not in sample_indices:
            continue
        image_name = row.get("image")
        instruction = row.get("editing_instruction")
        if not isinstance(image_name, str) or not image_name:
            raise ValueError("MIRA-Bench row has no valid image field")
        if not isinstance(instruction, str) or not instruction.strip():
            raise ValueError(f"MIRA-Bench instruction is missing for {image_name!r}")
        sample_id = f"mirage:{image_name}"
        if sample_id in seen_ids:
            raise ValueError(f"Duplicate MIRA-Bench sample ID: {sample_id}")
        seen_ids.add(sample_id)
        if requested_ids and sample_id not in requested_ids:
            continue
        safe_relative(image_name)
        selected_rows_by_index[row_index] = (
            sample_id,
            image_name,
            instruction.strip(),
        )

    selected_rows = (
        [
            selected_rows_by_index[index]
            for index in sample_indices
            if index in selected_rows_by_index
        ]
        if sample_indices
        else list(selected_rows_by_index.values())
    )
    matched_ids = {sample_id for sample_id, _, _ in selected_rows}
    missing_ids = requested_ids - matched_ids
    if missing_ids:
        raise ValueError(
            "Unknown MIRA-Bench sample ID(s): " + ", ".join(sorted(missing_ids))
        )
    missing_indices = set(sample_indices or []) - selected_rows_by_index.keys()
    if missing_indices:
        raise ValueError(
            "MIRA-Bench annotation indices are out of range: "
            + ", ".join(str(index) for index in sorted(missing_indices))
        )
    if (
        (sample_ids or sample_indices)
        and max_samples is not None
        and len(selected_rows) > max_samples
    ):
        raise ValueError(
            "max_samples would drop explicitly selected MIRA-Bench samples"
        )
    if max_samples is not None and not (sample_ids or sample_indices):
        selected_rows = selected_rows[:max_samples]

    samples: list[dict[str, Any]] = []
    for sample_id, image_name, instruction in selected_rows:
        downloaded_image = resolve_source_image(
            image_name,
            dataset_root=dataset_root,
        )
        cached_source = root / "source_images" / sample_key(sample_id) / "source.png"
        cached_source.parent.mkdir(parents=True, exist_ok=True)
        if (
            not cached_source.is_file()
            or cached_source.stat().st_mtime_ns < downloaded_image.stat().st_mtime_ns
        ):
            with Image.open(downloaded_image) as image:
                save_image(image, cached_source)
        samples.append(
            {
                "id": sample_id,
                "image": image_name,
                "instruction": instruction,
                "input_image": str(cached_source.relative_to(root)),
            }
        )
    if not samples:
        raise RuntimeError("No MIRA-Bench samples selected")
    root.mkdir(parents=True, exist_ok=True)
    selection_path = root / "selection.jsonl"
    selection_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in samples),
        encoding="utf-8",
    )
    return samples


def create_run_directory(output_root: Path) -> Path:
    output_root = output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_dir = output_root / f"run_{timestamp}"
    suffix = 1
    while run_dir.exists():
        run_dir = output_root / f"run_{timestamp}_{suffix:06d}"
        suffix += 1
    run_dir.mkdir()
    return run_dir


def resolve_run_directory(output_root: Path, run_name: str | None) -> Path:
    output_root = output_root.expanduser().resolve()
    if run_name is None or run_name == "latest":
        runs = sorted(
            path
            for path in output_root.glob("run_*")
            if path.is_dir() and (path / "selection.jsonl").is_file()
        )
        if not runs:
            raise FileNotFoundError(f"No MIRA-Bench runs found under {output_root}")
        return runs[-1]
    if Path(run_name).name != run_name or not run_name.startswith("run_"):
        raise ValueError("--run must be 'latest' or a run_<timestamp> directory name")
    run_dir = (output_root / run_name).resolve()
    if run_dir.parent != output_root or not (run_dir / "selection.jsonl").is_file():
        raise FileNotFoundError(f"MIRA-Bench run not found: {run_dir}")
    return run_dir


def result_paths(root: Path, model: str, sample: dict[str, Any]) -> tuple[Path, Path]:
    key = sample_key(sample["id"])
    return (
        root / "editing_results" / model / f"{model}_mira" / f"{key}.png",
        root / "records" / model / f"{key}.json",
    )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return _read_rows(path)


def write_summary(root: Path, samples: list[dict[str, Any]], models: list[str]) -> dict:
    rows: list[dict[str, Any]] = []
    counts: dict[str, Counter[str]] = {model: Counter() for model in models}
    for sample in samples:
        for model in models:
            output_path, record_path = result_paths(root, model, sample)
            record = read_json(record_path)
            status = record.get("status", "not_started")
            if status == "succeeded" and not png_is_valid(output_path):
                status = "missing_local_image"
            counts[model][status] += 1
            rows.append(
                {
                    "model": model,
                    "sample_id": sample["id"],
                    "image": sample["image"],
                    "instruction": sample["instruction"],
                    "status": status,
                    "prediction_id": record.get("prediction_id", ""),
                    "resolved_version": record.get("resolved_version", ""),
                    "output_path": (
                        str(output_path.relative_to(root))
                        if png_is_valid(output_path)
                        else ""
                    ),
                }
            )
    manifest_path = root / "inference_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {model: dict(counter) for model, counter in counts.items()}
    write_json(
        root / "summary.json",
        {
            "updated_at": utc_now(),
            "total_samples": len(samples),
            "models": summary,
        },
    )
    return summary


def process_one(
    root: Path,
    sample: dict[str, Any],
    model: str,
    *,
    client: Any,
    base_seed: int,
    poll_seconds: float,
    max_wait: float,
    bagel_thinking: bool,
    fast: bool,
    step1x_size_level: int,
    retry_failed: bool,
    limiter: SubmissionLimiter,
) -> str:
    output_path, record_path = result_paths(root, model, sample)
    record = read_json(record_path)
    seed = seed_for(sample["id"], base_seed)
    settings = model_input(
        model,
        sample["instruction"],
        "[local input image]",
        seed,
        bagel_thinking=bagel_thinking,
        fast=fast,
        step1x_size_level=step1x_size_level,
    )
    if record and (
        record.get("settings") != settings
        or record.get("prompt") != sample["instruction"]
    ):
        raise RuntimeError(
            f"Existing prediction settings differ for {model}/{sample['id']}. "
            "Use a new MIRA-Bench run directory for changed settings."
        )
    if record and record.get("model_spec") != MODELS[model]:
        raise RuntimeError(
            f"Existing Replicate model version differs for {model}/{sample['id']}"
        )
    if record.get("status") == "succeeded" and png_is_valid(output_path):
        return "skipped"
    if record.get("status") in {"failed", "canceled"} and not retry_failed:
        return "prior_failure"
    if record.get("status") in {"creating", "create_uncertain"} and not record.get(
        "prediction_id"
    ):
        if record.get("status") == "create_uncertain" and is_rejected_429(
            record.get("error", "")
        ):
            record["status"] = "rate_limited"
            write_json(record_path, record)
        else:
            return "create_uncertain"

    prediction_id = record.get("prediction_id")
    if not prediction_id or (
        record.get("status") in {"failed", "canceled"} and retry_failed
    ):
        source_path = root / sample["input_image"]
        with source_path.open("rb") as source_file:
            inputs = model_input(
                model,
                sample["instruction"],
                source_file,
                seed,
                bagel_thinking=bagel_thinking,
                fast=fast,
                step1x_size_level=step1x_size_level,
            )
            record = {
                "sample_id": sample["id"],
                "image": sample["image"],
                "model": model,
                "model_spec": MODELS[model],
                "prompt": sample["instruction"],
                "seed": seed,
                "settings": {
                    key: ("[local input image]" if hasattr(value, "read") else value)
                    for key, value in inputs.items()
                },
                "output_path": str(output_path.relative_to(root)),
                "status": "creating",
                "prediction_id": None,
                "submitted_at": utc_now(),
            }
            write_json(record_path, record)
            try:
                prediction = create_with_rate_limit_retry(
                    client, model, inputs, limiter
                )
            except Exception as exc:
                rejected = is_rejected_429(exc)
                record.update(
                    {
                        "status": "rate_limited" if rejected else "create_uncertain",
                        "error": str(exc),
                        "updated_at": utc_now(),
                    }
                )
                write_json(record_path, record)
                if rejected:
                    return "rate_limited"
                raise
        prediction_id = prediction.id
        record.update(
            {
                "status": prediction.status,
                "prediction_id": prediction_id,
                "resolved_version": getattr(prediction, "version", None),
                "updated_at": utc_now(),
            }
        )
        write_json(record_path, record)

    try:
        prediction = predict_until_done(
            client,
            prediction_id,
            poll_seconds=poll_seconds,
            max_wait=max_wait,
        )
        record.update(
            {
                "status": prediction.status,
                "resolved_version": getattr(prediction, "version", None),
                "prediction_error": getattr(prediction, "error", None),
                "updated_at": utc_now(),
            }
        )
        if prediction.status != "succeeded":
            record["error"] = str(getattr(prediction, "error", "Model failed"))
            write_json(record_path, record)
            return "failed"
        if prediction.output is None:
            raise RuntimeError(
                "Replicate output has expired; rerun with --retry-failed only after "
                "reviewing the record because a new prediction may incur a charge."
            )
        output_url = output_image_url(prediction.output)
        record["output_url"] = output_url
        write_json(record_path, record)
        width, height = download_result_image(output_url, output_path)
        record.update(
            {
                "status": "succeeded",
                "image_size": [width, height],
                "saved_at": utc_now(),
            }
        )
        write_json(record_path, record)
        return "succeeded"
    except Exception as exc:
        record.update({"last_retrieval_error": repr(exc), "updated_at": utc_now()})
        if record.get("status") not in {"failed", "canceled", "succeeded"}:
            record["status"] = "pending_retrieval"
        write_json(record_path, record)
        raise


def run_benchmark(
    args: argparse.Namespace,
    *,
    rows: list[dict[str, Any]] | None = None,
    dataset_root: Path | None = None,
    client: Any = None,
) -> dict:
    output_root = Path(args.output_root).expanduser().resolve()
    if args.resume:
        root = resolve_run_directory(output_root, args.run)
    else:
        if args.run:
            raise ValueError("--run requires --resume")
        root = create_run_directory(output_root)
    if rows is None:
        rows, detected_root = load_dataset_rows(
            Path(args.dataset_dir) if args.dataset_dir else None
        )
        if dataset_root is None:
            dataset_root = detected_root
    if not args.dry_run and client is None:
        if not os.environ.get("REPLICATE_API_TOKEN"):
            raise RuntimeError("Set REPLICATE_API_TOKEN or use --dry-run.")
        import replicate

        client = replicate.Client(api_token=os.environ["REPLICATE_API_TOKEN"])

    samples = prepare_samples(
        root,
        rows,
        dataset_root=dataset_root,
        sample_ids=args.sample_ids,
        sample_indices=args.sample_indices,
        max_samples=args.max_samples,
    )
    print(f"Prepared {len(samples)} MIRA-Bench samples.", flush=True)
    config = {
        "dataset": DATASET_ID,
        "split": "benchmark",
        "sample_ids": args.sample_ids,
        "sample_indices": args.sample_indices,
        "models": args.models,
        "model_specs": {model: MODELS[model] for model in args.models},
        "base_seed": args.seed,
        "workers": args.workers,
        "updated_at": utc_now(),
    }
    if args.dry_run:
        write_json(root / "run_config.json", config)
        print("DRY RUN: no Replicate predictions created.")
        print(f"Output: {root}")
        return write_summary(root, samples, args.models)

    write_json(root / "run_config.json", config)
    limiter = SubmissionLimiter(args.submit_interval)
    jobs = [(sample, model) for sample in samples for model in args.models]
    pending = {}
    iterator = iter(jobs)
    total = len(jobs)
    done = 0

    def process(sample: dict[str, Any], model: str) -> str:
        return process_one(
            root,
            sample,
            model,
            client=client,
            base_seed=args.seed,
            poll_seconds=args.poll_seconds,
            max_wait=args.max_wait,
            bagel_thinking=args.bagel_thinking,
            fast=args.fast,
            step1x_size_level=args.step1x_size_level,
            retry_failed=args.retry_failed,
            limiter=limiter,
        )

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        for _ in range(min(args.workers * 2, total)):
            try:
                sample, model = next(iterator)
            except StopIteration:
                break
            pending[executor.submit(process, sample, model)] = (sample, model)
        while pending:
            finished, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in finished:
                sample, model = pending.pop(future)
                try:
                    status = future.result()
                except Exception as exc:
                    status = "error"
                    print(
                        f"{model} {sample['id']}: ERROR: {exc}",
                        flush=True,
                    )
                    if args.stop_on_error:
                        for other in pending:
                            other.cancel()
                        write_summary(root, samples, args.models)
                        raise
                done += 1
                print(f"[{done}/{total}] {model} {sample['id']}: {status}", flush=True)
                if args.sleep_seconds > 0 and status == "succeeded":
                    time.sleep(args.sleep_seconds)
                try:
                    next_sample, next_model = next(iterator)
                except StopIteration:
                    continue
                pending[executor.submit(process, next_sample, next_model)] = (
                    next_sample,
                    next_model,
                )
    summary = write_summary(root, samples, args.models)
    print(f"Done: {json.dumps(summary, indent=2)}")
    print(f"Results in: {root}")
    return summary


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run selected MIRA-Bench image edits through Replicate."
    )
    parser.add_argument(
        "--output-root",
        default=str(DEFAULT_OUTPUT_ROOT),
        help="Parent directory for timestamped run_<timestamp> output directories.",
    )
    parser.add_argument(
        "--dataset-dir",
        help="Optional local MIRA-Bench dataset root or its benchmark/ directory.",
    )
    parser.add_argument(
        "--sample-ids",
        nargs="+",
        help="MIRA-Bench sample IDs such as mirage:0001.jpg. Default: all samples.",
    )
    parser.add_argument(
        "--sample-indices",
        nargs="+",
        type=int,
        help="Zero-based positions in benchmark/annotations.jsonl.",
    )
    parser.add_argument(
        "--models", nargs="+", default=list(MODELS), choices=list(MODELS)
    )
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume the latest run, or the run selected by --run.",
    )
    parser.add_argument(
        "--run",
        help="Existing run_<timestamp> directory name to resume.",
    )
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--bagel-thinking", action="store_true")
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--step1x-size-level", type=int, default=512)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--submit-interval", type=float, default=1.0)
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    parser.add_argument("--max-wait", type=float, default=2400.0)
    parser.add_argument("--sleep-seconds", type=float, default=0.0)
    parser.add_argument("--stop-on-error", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = make_parser().parse_args(argv)
    if args.max_samples is not None and args.max_samples <= 0:
        raise SystemExit("--max-samples must be >= 1")
    if args.sample_ids:
        if len(set(args.sample_ids)) != len(args.sample_ids):
            raise SystemExit("--sample-ids must not contain duplicates")
    if args.sample_indices:
        if len(set(args.sample_indices)) != len(args.sample_indices):
            raise SystemExit("--sample-indices must not contain duplicates")
        if any(index < 0 for index in args.sample_indices):
            raise SystemExit("--sample-indices must be non-negative")
    if args.sample_ids and args.sample_indices:
        raise SystemExit("--sample-ids and --sample-indices are mutually exclusive")
    if args.step1x_size_level <= 0 or args.poll_seconds <= 0 or args.max_wait <= 0:
        raise SystemExit("Size level, poll seconds, and max wait must be positive")
    if args.workers < 1 or args.submit_interval < 0 or args.sleep_seconds < 0:
        raise SystemExit("--workers must be >= 1; timing values cannot be negative")
    if args.run and not args.resume:
        raise SystemExit("--run can only be used with --resume")
    run_benchmark(args)


if __name__ == "__main__":
    main()
