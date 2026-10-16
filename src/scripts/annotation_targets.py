from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from framework.utils.project import load_project_environment

ROOT = Path(__file__).resolve().parents[2]
SUPPORTED_BENCHMARKS = ("compbench", "magicbrush", "refedit_bench", "mira_bench")


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
    elif benchmark == "mira_bench":
        success_name = "mira_bench_mixed.jsonl"
        audit_name = "mira_bench_audit.jsonl"
        directory = "mira_bench"
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
    if benchmark == "mira_bench":
        run_dirs = sorted(path for path in output_dir.glob("run_*") if path.is_dir())
        if run_dirs:
            output_dir = run_dirs[-1]
        elif not configured_output:
            legacy_root = annotation_root / "mirage"
            legacy_run_dirs = sorted(
                path
                for path in legacy_root.glob("run_*")
                if path.is_dir() and (path / "mirage_mixed.jsonl").is_file()
            )
            if legacy_run_dirs:
                output_dir = legacy_run_dirs[-1]
                success_name = "mirage_mixed.jsonl"
                audit_name = "mirage_audit.jsonl"

    return AnnotationTarget(
        annotations=output_dir / success_name,
        audit=output_dir / audit_name,
    )
