from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from PIL import Image

from annotation.parser import GoogleParser
from annotation.refine import (
    refine_annotations,
    refined_audit_path,
    refined_output_path,
)
from evaluation.common import HFImageLoader
from framework.inference.config import InferenceConfig
from framework.inference.pool import ModelPool
from framework.inference.sam3 import sam3_segment_masks
from framework.inference.utils import get_device
from framework.utils.project import load_project_environment
from framework.utils.randomness import seed_random_generators

if __package__:
    from .annotation_targets import ROOT, annotation_target
else:
    from annotation_targets import ROOT, annotation_target


def resolve_annotations_path(path: Path | None, *, default: Path) -> Path:
    """Accept a JSONL file or an annotation run directory."""
    resolved = (path or default).expanduser().resolve()
    if resolved.is_dir():
        candidate = resolved / "refedit_bench_mixed.jsonl"
        if candidate.is_file():
            return candidate
        mixed = sorted(
            item
            for item in resolved.glob("*mixed.jsonl")
            if not item.name.startswith("refined_")
        )
        if len(mixed) == 1:
            return mixed[0]
        raise FileNotFoundError(
            f"No refedit_bench_mixed.jsonl in annotation directory: {resolved}"
        )
    if not resolved.is_file():
        raise FileNotFoundError(f"Annotation JSONL not found: {resolved}")
    return resolved


def _sam3_pool() -> ModelPool:
    """Load SAM 3 without constructing the full Klein ModelPool."""
    pool = ModelPool.__new__(ModelPool)
    seed_random_generators()
    pool.config = InferenceConfig()
    pool.device = get_device()
    return pool


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Review RefEdit mixed annotations and, when the dataset mask covers "
            "a host instead of the edited part, re-segment with SAM 3 text mode. "
            "Writes refined_<name>.jsonl beside the input and stores new masks "
            "under assets/refined_<stem>/."
        )
    )
    parser.add_argument(
        "--annotations",
        "--annotation",
        dest="annotations",
        type=Path,
        help=(
            "Mixed annotation JSONL, or a run directory containing "
            "refedit_bench_mixed.jsonl (defaults to the latest RefEdit output)."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Refined JSONL path (defaults to refined_<annotations name>).",
    )
    parser.add_argument(
        "--audit",
        type=Path,
        help="Audit JSONL for failed refinements.",
    )
    parser.add_argument(
        "--sample-id",
        action="append",
        default=[],
        help="Limit to these sample_ids (repeatable).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of samples to review.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Ask the LLM but do not run SAM 3 or write files.",
    )
    args = parser.parse_args()

    load_project_environment(ROOT)
    target = annotation_target("refedit_bench", ROOT)
    annotations_path = resolve_annotations_path(
        args.annotations, default=target.annotations
    )

    output_path = (
        args.output.expanduser().resolve()
        if args.output
        else refined_output_path(annotations_path)
    )
    audit_path = (
        args.audit.expanduser().resolve()
        if args.audit
        else refined_audit_path(annotations_path)
    )

    image_loader = HFImageLoader()
    sam3 = None

    def load_image(record: dict[str, Any]) -> Image.Image:
        source, _ = image_loader.load_pair(record)
        return source

    def segment_text(image: Image.Image, text: str) -> list[Image.Image]:
        nonlocal sam3
        if sam3 is None:
            sam3 = _sam3_pool()
        masks = sam3_segment_masks(sam3, image, text)
        if not masks:
            raise ValueError(f"No object matched the text description: {text!r}")
        return masks

    counts = refine_annotations(
        annotations_path,
        parser=GoogleParser(),
        load_image=load_image,
        segment_text=segment_text,
        output_path=output_path,
        audit_path=audit_path,
        sample_ids=set(args.sample_id) or None,
        limit=args.limit,
        dry_run=args.dry_run,
    )
    action = "Would write" if args.dry_run else "Wrote"
    print(f"{action} refined annotations: {output_path}")
    print(
        f"records={counts['records']} copied={counts['copied']} "
        f"optimized={counts['optimized']} failed={counts['failed']} "
        f"resumed={counts['resumed']}"
    )
    if not args.dry_run:
        print(f"Audit: {audit_path}")


if __name__ == "__main__":
    main()
