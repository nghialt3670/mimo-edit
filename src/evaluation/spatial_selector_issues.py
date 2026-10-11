from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .common import read_jsonl
from .notebook import resolve_run_directory
from .spatial import needs_spatial_anchor_box


def find_spatial_selector_issues(run_dir: Path) -> list[dict[str, Any]]:
    """Find completed mixed-mode samples using a point/scribble for spatial placement."""
    run_dir = run_dir.expanduser().resolve()
    results_path = run_dir / "results" / "results.jsonl"
    if not results_path.is_file():
        raise FileNotFoundError(f"Inference results not found: {results_path}")

    issues = []
    for result in read_jsonl(results_path):
        instruction = result.get("instruction")
        if (
            result.get("status") != "completed"
            or result.get("prompt_mode") != "mixed"
            or not isinstance(instruction, str)
            or not needs_spatial_anchor_box(instruction)
        ):
            continue

        raw_path_value = result.get("raw_path")
        if not isinstance(raw_path_value, str):
            raise ValueError(
                f"Missing raw_path for annotation {result.get('annotation_index')!r} "
                f"in {results_path}"
            )
        raw_path = Path(raw_path_value)
        if not raw_path.is_absolute():
            raw_path = run_dir / raw_path
        if not raw_path.is_file():
            raise FileNotFoundError(f"Raw inference record not found: {raw_path}")
        raw = json.loads(raw_path.read_text(encoding="utf-8"))
        reference_record = raw.get("reference_record") or {}
        references = {
            reference.get("token"): reference
            for reference in reference_record.get("references", [])
        }

        matched_references = []
        for selected in raw.get("reference_prompts", []):
            if selected.get("type") not in ("point", "scribble"):
                continue
            reference = references.get(selected.get("token"))
            visual_prompts = (reference or {}).get("visual_prompts") or {}
            box = (visual_prompts.get("box") or {}).get("xyxy")
            if box is not None:
                matched_references.append(
                    {
                        "token": selected.get("token"),
                        "selected_type": selected["type"],
                        "box": box,
                    }
                )

        if matched_references:
            issues.append(
                {
                    "annotation_index": result.get("annotation_index"),
                    "sample_id": result.get("sample_id"),
                    "instruction": instruction,
                    "references": matched_references,
                    "raw_path": str(raw_path),
                }
            )
    return issues


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Find completed mixed-mode spatial-placement samples that used point or "
            "scribble selectors despite having a full reference box."
        )
    )
    parser.add_argument(
        "output_root",
        type=Path,
        help="Prompt-mode output directory containing run_* folders, or a run directory.",
    )
    parser.add_argument(
        "--run",
        default="latest",
        help="Run directory name under output_root (default: latest).",
    )
    parser.add_argument(
        "--indices-only",
        action="store_true",
        help="Print only the JSON array of annotation indices.",
    )
    args = parser.parse_args()

    if (args.output_root / "results" / "results.jsonl").is_file():
        run_dir = args.output_root.expanduser().resolve()
    else:
        run_dir = resolve_run_directory(args.output_root, args.run)
    issues = find_spatial_selector_issues(run_dir)
    if args.indices_only:
        print(json.dumps([issue["annotation_index"] for issue in issues]))
        return
    print(
        json.dumps(
            {
                "run_dir": str(run_dir),
                "annotation_indices": [issue["annotation_index"] for issue in issues],
                "issues": issues,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
