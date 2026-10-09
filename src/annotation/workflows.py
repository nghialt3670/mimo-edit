from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

from framework.utils.project import load_project_environment

from .core import Annotator
from .grounder import GroundingDinoGrounder, HybridGrounder, MaskOnlyGrounder
from .parser import GoogleParser
from .runner import Adapter, convert_rows


@dataclass(frozen=True)
class AnnotationConfig:
    output_dir: Path
    limit: int | None = 10
    streaming: bool = True
    model: str = "google/gemini-2.5-flash-lite"
    grounder_model: str = "IDEA-Research/grounding-dino-tiny"
    box_threshold: float = 0.35
    text_threshold: float = 0.25
    hf_token: str | None = None

    @classmethod
    def from_env(
        cls,
        benchmark: str,
    ) -> AnnotationConfig:
        project_root = load_project_environment()
        default_directories = {
            "compbench": "compbench",
            "magicbrush": "magicbrush",
            "refedit_bench": "refedit_bench",
        }
        try:
            output_directory = default_directories[benchmark]
        except KeyError as exc:
            raise ValueError(f"Unsupported annotation benchmark: {benchmark!r}") from exc
        raw_limit = int(os.environ.get("ANNOTATE_LIMIT", "10"))
        if raw_limit < 0:
            raise ValueError("ANNOTATE_LIMIT must be zero or a positive integer.")
        box_threshold = float(os.environ.get("ANNOTATE_BOX_THRESHOLD", "0.35"))
        text_threshold = float(os.environ.get("ANNOTATE_TEXT_THRESHOLD", "0.25"))
        if not 0 <= box_threshold <= 1 or not 0 <= text_threshold <= 1:
            raise ValueError("Grounding thresholds must be between 0 and 1.")
        configured_output = os.environ.get("ANNOTATE_OUT")
        output_dir = (
            Path(configured_output).expanduser()
            if configured_output
            else project_root / ".out" / "annotation" / output_directory
        )
        return cls(
            output_dir=output_dir.resolve(),
            limit=None if raw_limit == 0 else raw_limit,
            streaming=os.environ.get("ANNOTATE_STREAMING", "1") != "0",
            model=os.environ.get("ANNOTATE_LLM", "google/gemini-2.5-flash-lite"),
            grounder_model=os.environ.get(
                "ANNOTATE_GROUNDER", "IDEA-Research/grounding-dino-tiny"
            ),
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            hf_token=os.environ.get("HF_TOKEN") or None,
        )

    def create_annotator(self, *, use_grounder: bool) -> Annotator:
        parser = GoogleParser(model=self.model)
        if use_grounder:
            grounder = HybridGrounder(
                GroundingDinoGrounder(
                    model_id=self.grounder_model,
                    threshold=self.box_threshold,
                    text_threshold=self.text_threshold,
                ),
                prefer_benchmark_mask=True,
            )
        else:
            grounder = MaskOnlyGrounder()
        return Annotator(parser=parser, grounder=grounder)


def comma_separated_env(name: str, default: str) -> set[str]:
    """Read a comma-separated setting as a normalized set."""
    return {
        item.strip()
        for item in os.environ.get(name, default).split(",")
        if item.strip()
    }


def selected_rows(
    rows: Iterable[dict[str, Any]],
    predicate: Callable[[dict[str, Any]], bool],
) -> Iterable[dict[str, Any]]:
    """Filter dataset rows while preserving their original Hugging Face indices."""
    for row_index, row in enumerate(rows):
        if predicate(row):
            selected = dict(row)
            selected["_row_index"] = row_index
            yield selected


def run_hf_annotation(
    *,
    dataset_id: str,
    split: str,
    adapter: Adapter,
    config: AnnotationConfig,
    output_name: str,
    audit_name: str,
    expected_columns: set[str],
    use_grounder: bool,
    row_filter: Callable[[dict[str, Any]], bool] | None = None,
) -> dict[str, int]:
    """Load a dataset split, validate its adapter contract, and annotate rows."""
    from datasets import load_dataset

    dataset = load_dataset(
        dataset_id,
        split=split,
        streaming=config.streaming,
        token=config.hf_token,
    )
    columns = set(dataset.column_names)
    missing = expected_columns - columns
    if missing:
        raise ValueError(
            f"{dataset_id}/{split} is missing columns {sorted(missing)}; "
            f"available columns are {sorted(columns)}."
        )
    config.output_dir.mkdir(parents=True, exist_ok=True)
    rows = selected_rows(dataset, row_filter) if row_filter is not None else dataset
    return convert_rows(
        rows=rows,
        adapter=adapter,
        split=split,
        annotator=config.create_annotator(use_grounder=use_grounder),
        output=config.output_dir / output_name,
        failures=config.output_dir / audit_name,
        limit=config.limit,
    )
