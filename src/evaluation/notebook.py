from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from framework.utils.project import load_project_environment
from framework.utils.randomness import seed_random_generators

from .common import HFImageLoader, read_jsonl

PromptMode = Literal["text", "mixed"]

_BENCHMARKS = {
    "compbench": (
        "MMEDIT_COMPBENCH_ANNOTATIONS",
        "MMEDIT_COMPBENCH_ANNOTATION_RUN",
        "compbench_local_multiobject_mixed.jsonl",
        "MMEDIT_COMPBENCH_OUTPUT_DIR",
    ),
    "refedit_bench": (
        "MMEDIT_REFEDIT_BENCH_ANNOTATIONS",
        "MMEDIT_REFEDIT_BENCH_ANNOTATION_RUN",
        "refedit_bench_mixed.jsonl",
        "MMEDIT_REFEDIT_BENCH_OUTPUT_DIR",
    ),
}


@dataclass
class BenchmarkContext:
    project_root: Path
    annotation_file: Path
    output_dir: Path
    records: list[dict[str, Any]]
    image_loader: HFImageLoader


def resolve_annotation_file(
    *,
    project_root: Path,
    benchmark: str,
    annotation_override_env: str,
    run_selection_env: str,
    filename: str,
) -> Path:
    """Resolve an explicit annotation file or the selected timestamped run."""
    configured_file = os.environ.get(annotation_override_env)
    if configured_file:
        annotation_file = Path(configured_file).expanduser()
        if not annotation_file.is_absolute():
            annotation_file = project_root / annotation_file
        annotation_file = annotation_file.resolve()
        if not annotation_file.is_file():
            raise FileNotFoundError(
                f"{benchmark} annotations not found: {annotation_file}."
            )
        return annotation_file

    annotation_root = Path(
        os.environ.get("ANNOTATE_OUT")
        or project_root / ".out" / "annotation" / benchmark
    ).expanduser()
    if not annotation_root.is_absolute():
        annotation_root = project_root / annotation_root
    annotation_root = annotation_root.resolve()

    selected_run = os.environ.get(run_selection_env) or "latest"
    if selected_run == "latest":
        run_directories = sorted(
            path for path in annotation_root.glob("run_*") if path.is_dir()
        )
        available_runs = [
            path for path in run_directories if (path / filename).is_file()
        ]
        if available_runs:
            return available_runs[-1] / filename
        if run_directories:
            raise FileNotFoundError(
                f"No {filename} found in annotation runs under {annotation_root}."
            )
        legacy_file = annotation_root / filename
        if legacy_file.is_file():
            return legacy_file
        raise FileNotFoundError(
            f"No {benchmark} annotation run containing {filename} found under "
            f"{annotation_root}. Run the annotation notebook first or set "
            f"{annotation_override_env}."
        )

    if Path(selected_run).name != selected_run or not selected_run.startswith("run_"):
        raise ValueError(
            f"{run_selection_env} must be 'latest' or a run_<timestamp> directory name."
        )
    selected_file = (annotation_root / selected_run / filename).resolve()
    if selected_file.parent.parent != annotation_root or not selected_file.is_file():
        raise FileNotFoundError(
            f"{benchmark} annotations not found in selected run: {selected_file}."
        )
    return selected_file


def load_benchmark(
    benchmark: str,
    prompt_mode: PromptMode = "mixed",
) -> BenchmarkContext:
    """Load common local paths and annotations for a supported benchmark."""
    project_root = load_project_environment()
    seed_random_generators()
    if prompt_mode not in ("text", "mixed"):
        raise ValueError(f"Unsupported prompt mode: {prompt_mode!r}")
    try:
        annotation_env, run_env, filename, output_env = _BENCHMARKS[benchmark]
    except KeyError as exc:
        raise ValueError(f"Unsupported benchmark: {benchmark!r}") from exc

    annotation_file = resolve_annotation_file(
        project_root=project_root,
        benchmark=benchmark,
        annotation_override_env=annotation_env,
        run_selection_env=run_env,
        filename=filename,
    )
    output_dir = (
        Path(
            os.environ.get(output_env)
            or str(project_root / ".out" / "evaluation" / benchmark)
        )
        .expanduser()
        .resolve()
    )
    output_dir = output_dir / prompt_mode
    records = list(read_jsonl(annotation_file))
    if not records:
        raise ValueError(f"No {benchmark} annotations found in {annotation_file}")
    return BenchmarkContext(
        project_root=project_root,
        annotation_file=annotation_file,
        output_dir=output_dir,
        records=records,
        image_loader=HFImageLoader(),
    )


def create_run_directory(output_root: Path) -> Path:
    """Create a unique timestamped directory for one inference run."""
    output_root = output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    suffix = 1
    while True:
        suffix_text = "" if suffix == 1 else f"_{suffix:06d}"
        run_dir = output_root / f"run_{timestamp}{suffix_text}"
        try:
            run_dir.mkdir()
        except FileExistsError:
            suffix += 1
            continue
        return run_dir


def resolve_run_directory(
    output_root: Path,
    run_name: str | None = None,
) -> Path:
    """Resolve a selected inference run, defaulting to the latest run."""
    output_root = output_root.expanduser().resolve()
    if run_name == "latest" or run_name is None:
        run_dirs = sorted(path for path in output_root.glob("run_*") if path.is_dir())
        runs = [
            path for path in run_dirs if (path / "results" / "results.jsonl").is_file()
        ]
        if runs:
            return runs[-1]
        if run_dirs:
            raise FileNotFoundError(
                f"No completed inference results found in runs under {output_root}."
            )
        if (output_root / "results" / "results.jsonl").is_file():
            return output_root
        raise FileNotFoundError(
            f"No inference runs found under {output_root}. Run the inference notebook first."
        )

    if Path(run_name).name != run_name or not run_name.startswith("run_"):
        raise ValueError(
            "run_name must be 'latest' or a run_<timestamp> directory name."
        )
    run_dir = (output_root / run_name).resolve()
    if run_dir.parent != output_root:
        raise ValueError(
            f"Run directory must be directly under {output_root}: {run_name!r}"
        )
    if not (run_dir / "results" / "results.jsonl").is_file():
        raise FileNotFoundError(
            f"Inference results not found in run directory: {run_dir}"
        )
    return run_dir


def create_agent_model() -> Any:
    """Create the shared OpenAI-compatible model configured for inference."""
    load_project_environment()
    seed_random_generators()
    api_key = os.environ.get("MMEDIT_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise EnvironmentError("Set MMEDIT_API_KEY or GOOGLE_API_KEY.")

    from smolagents import OpenAIModel

    return OpenAIModel(
        model_id=os.environ.get("MMEDIT_MODEL_ID") or "gemini-2.5-flash",
        api_base=os.environ.get("MMEDIT_OPENAI_BASE_URL")
        or "https://generativelanguage.googleapis.com/v1beta/openai/",
        api_key=api_key,
        temperature=0,
    )
