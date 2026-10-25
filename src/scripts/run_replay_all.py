"""Run the four first-turn replay inference jobs in one process with one shared ModelPool.

Extra arguments are forwarded to every job, for example:
  uv run python src/scripts/run_replay_all.py --limit 2
"""

from __future__ import annotations

import importlib
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

JOBS = (
    ("compbench_mixed", "compbench_inference_mixed", "resources/compbench_inference_mixed_first_turn.jsonl"),
    ("compbench_text", "compbench_inference_text", "resources/compbench_inference_text_first_turn.jsonl"),
    ("refedit_mixed", "refedit_bench_inference_mixed", "resources/refedit_bench_inference_mixed_first_turn.jsonl"),
    ("refedit_text", "refedit_bench_inference_text", "resources/refedit_bench_inference_text_first_turn.jsonl"),
)


def main() -> None:
    from framework.inference import InferenceConfig, ModelPool
    from framework.utils.project import load_project_environment

    load_project_environment()
    extra = sys.argv[1:]
    pool = ModelPool(InferenceConfig())

    failed: list[str] = []
    for name, module_name, store in JOBS:
        print(f"\n=== {name} ===")
        module = importlib.import_module(module_name)
        argv = ["--first-turn", "replay", "--first-turn-file", str(ROOT / store), *extra]
        try:
            module.main(argv, pool=pool)
        except SystemExit as exc:
            print(f"{name}: exited with {exc.code}")
            failed.append(name)
        except Exception:
            traceback.print_exc()
            failed.append(name)

    print("\nSummary:")
    for name, _, _ in JOBS:
        print(f"  {name}: {'failed' if name in failed else 'done'}")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
