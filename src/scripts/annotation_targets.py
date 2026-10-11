from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from framework.utils.project import load_project_environment

ROOT = Path(__file__).resolve().parents[2]
SUPPORTED_BENCHMARKS = ("compbench", "magicbrush", "refedit_bench")


@dataclass(frozen=True)
class AnnotationTarget:
    annotations: Path
    audit: Path


def annotation_target(benchmark: str, root: Path = ROOT) -> AnnotationTarget:
    """Resolve the default success and audit files for a benchmark."""
    if benchmark not in SUPPORTED_BENCHMARKS:
        raise ValueError(
            f"Unsupported benchmark {benchmark!r}; choose from "
            f"{', '.join(SUPPORTED_BENCHMARKS)}."
        )

    load_project_environment(root)
    annotation_root = root / ".out" / "annotation"
    if benchmark == "compbench":
        success_name = "compbench_local_multiobject_mixed.jsonl"
        audit_name = "compbench_local_multiobject_audit.jsonl"
        directory = "compbench"
    elif benchmark == "magicbrush":
        split = os.environ.get("MAGICBRUSH_SPLIT") or "dev"
        if split not in {"train", "dev"}:
            raise ValueError("MAGICBRUSH_SPLIT must be 'train' or 'dev'.")
        success_name = f"magicbrush_{split}_mixed.jsonl"
        audit_name = f"magicbrush_{split}_audit.jsonl"
        directory = "magicbrush"
    else:
        success_name = "refedit_bench_mixed.jsonl"
        audit_name = "refedit_bench_audit.jsonl"
        directory = "refedit_bench"

    configured_output = os.environ.get("ANNOTATE_OUT")
    output_dir = (
        Path(configured_output).expanduser()
        if configured_output
        else annotation_root / directory
    )
    return AnnotationTarget(
        annotations=output_dir / success_name,
        audit=output_dir / audit_name,
    )
