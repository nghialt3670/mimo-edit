"""Saved first-turn agent commands for benchmark inference.

A first-turn command is the single code action the agent produces for the
opening user message. Generating it once and replaying it in later runs avoids
prompting the model again for the same sample.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

FIRST_TURN_FILE_NAME = "first_turn.jsonl"
_PROMPT_TOKEN = re.compile(r"<(box|point|scribble)_\d+>")


def task_fingerprint(task: str, objects: dict[str, Any]) -> str:
    """Hash the task text and the object names the first turn was generated for."""
    payload = json.dumps(
        {"task": task, "objects": sorted(objects)},
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def prompt_signature(prompt_types: Iterable[str]) -> str:
    """Join selected visual prompt types in reference order, e.g. 'box,point'."""
    return ",".join(prompt_types)


def prompt_signature_from_task(task: str) -> str:
    """Recover the prompt signature from the prompt tokens in a task string."""
    return prompt_signature(_PROMPT_TOKEN.findall(task))


@dataclass(frozen=True, slots=True)
class FirstTurnEntry:
    sample_id: str
    prompt_mode: str
    prompt_signature: str
    fingerprint: str
    task: str
    objects: list[str]
    thought: str
    code: str
    raw_output: str
    model_id: str | None


class FirstTurnStore:
    """Append-only JSONL store keyed by sample id, prompt mode, and prompt signature.

    The last entry wins. Lines without a prompt_signature (legacy) are ignored.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self._entries: dict[tuple[str, str, str], FirstTurnEntry] = {}
        if path.is_file():
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                record = json.loads(line)
                if "prompt_signature" not in record:
                    continue
                entry = FirstTurnEntry(**record)
                self._entries[self._key(entry)] = entry

    @staticmethod
    def _key(entry: FirstTurnEntry) -> tuple[str, str, str]:
        return (entry.sample_id, entry.prompt_mode, entry.prompt_signature)

    def get(
        self, sample_id: str, prompt_mode: str, signature: str
    ) -> FirstTurnEntry | None:
        return self._entries.get((sample_id, prompt_mode, signature))

    def put(self, entry: FirstTurnEntry) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(asdict(entry), ensure_ascii=False) + "\n")
        self._entries[self._key(entry)] = entry


@dataclass(frozen=True, slots=True)
class FirstTurnConfig:
    mode: Literal["generate", "replay"]
    store: FirstTurnStore


def add_first_turn_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--first-turn",
        choices=["off", "generate", "replay"],
        default="off",
        help=(
            "generate: ask the agent for the first code action only and save it, no execution. "
            "replay: execute the saved first action and continue the ReAct loop only on error."
        ),
    )
    parser.add_argument(
        "--first-turn-file",
        type=Path,
        default=None,
        help=f"JSONL file for saved first-turn commands. Defaults to <output root>/{FIRST_TURN_FILE_NAME}.",
    )
    parser.add_argument(
        "--from-run",
        type=Path,
        default=None,
        help=(
            "Existing run_<timestamp> directory. Re-executes each sample's saved act code "
            "without model calls, falling back to the model only when a code action fails."
        ),
    )


def build_from_run(args: argparse.Namespace) -> Path | None:
    if args.from_run is None:
        return None
    if args.first_turn != "off":
        raise ValueError("--from-run cannot be combined with --first-turn.")
    if getattr(args, "resume", False):
        raise ValueError("--from-run cannot be combined with --resume.")
    run_dir = args.from_run.expanduser().resolve()
    if not (run_dir / "logs").is_dir():
        raise FileNotFoundError(f"--from-run has no logs directory: {run_dir}")
    print(f"Re-executing saved code from: {run_dir}")
    return run_dir


def build_first_turn_config(
    args: argparse.Namespace,
    output_root: Path,
) -> FirstTurnConfig | None:
    if args.first_turn == "off":
        if args.first_turn_file is not None:
            raise ValueError("--first-turn-file requires --first-turn generate or replay.")
        return None
    path = (args.first_turn_file or output_root / FIRST_TURN_FILE_NAME).expanduser().resolve()
    print(f"First-turn mode: {args.first_turn} ({path})")
    return FirstTurnConfig(mode=args.first_turn, store=FirstTurnStore(path))


def _file_stem(sample_id: str, annotation_index: int) -> str:
    # Mirrors evaluation.common._file_stem; results.jsonl log_dir can be stale.
    safe_id = re.sub(r"[^A-Za-z0-9._-]+", "_", str(sample_id)).strip("._") or "sample"
    return f"{annotation_index:06d}_{safe_id}"


def _find_log_dir(run_dir: Path, sample_id: str, annotation_index: int) -> Path | None:
    # Folder index can differ from the recorded annotation_index; fall back to a unique name match.
    logs = run_dir / "logs"
    exact = logs / _file_stem(sample_id, annotation_index)
    if exact.is_dir():
        return exact
    safe_id = _file_stem(sample_id, 0).split("_", 1)[1]
    matches = sorted(logs.glob(f"*_{safe_id}"))
    return matches[0] if len(matches) == 1 else None


def _first_model_output(raw_steps: list[Any]) -> str | None:
    for step in raw_steps:
        output = step.get("model_output") if isinstance(step, dict) else None
        if isinstance(output, str) and output.strip():
            return output
    return None


def export_run_to_first_turn(
    run_dir: Path, store_path: Path | None = None
) -> tuple[Path, int, list[str]]:
    """Write each sample's first code action from a finished run into a first-turn store.

    Task, objects, and prompt types come from results/results.jsonl. The first act
    and its thought come from logs/<sample>/trajectory.json; the raw model output and
    model id come from logs/<sample>/raw.json when present.

    Returns the store path, the number of exported entries, and the skipped sample ids.
    """
    run_dir = run_dir.expanduser().resolve()
    results_path = run_dir / "results" / "results.jsonl"
    if not results_path.is_file():
        raise FileNotFoundError(f"No results file: {results_path}")
    path = (store_path or run_dir / FIRST_TURN_FILE_NAME).expanduser().resolve()
    store = FirstTurnStore(path)

    latest: dict[str, dict[str, Any]] = {}
    for line in results_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            record = json.loads(line)
            latest[str(record.get("sample_id"))] = record

    exported = 0
    skipped: list[str] = []
    for sample_id, record in latest.items():
        log_dir = _find_log_dir(run_dir, sample_id, int(record["annotation_index"]))
        trajectory_path = log_dir / "trajectory.json" if log_dir else None
        if trajectory_path is None or not trajectory_path.is_file():
            skipped.append(sample_id)
            continue
        steps = json.loads(trajectory_path.read_text(encoding="utf-8"))
        first = next((step for step in steps if "act" in step), None)
        if first is None:
            skipped.append(sample_id)
            continue

        code = str(first["act"])
        thought = str(first.get("thought") or "")
        raw_output = code
        model_id = None
        raw_path = log_dir / "raw.json"
        if raw_path.is_file():
            raw = json.loads(raw_path.read_text(encoding="utf-8"))
            model_id = raw.get("model_id")
            raw_output = _first_model_output(raw.get("agent_raw_steps") or []) or code

        references = record.get("reference_prompts") or []
        objects = ["image_0", *(ref["variable"] for ref in references)]
        task = str(record["instruction"])
        store.put(
            FirstTurnEntry(
                sample_id=sample_id,
                prompt_mode=str(record["prompt_mode"]),
                prompt_signature=prompt_signature(ref["type"] for ref in references),
                fingerprint=task_fingerprint(task, dict.fromkeys(objects)),
                task=task,
                objects=sorted(objects),
                thought=thought,
                code=code,
                raw_output=raw_output,
                model_id=model_id,
            )
        )
        exported += 1
    return path, exported, skipped


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export a finished run's first code actions to a first-turn JSONL store."
    )
    parser.add_argument("--run-dir", type=Path, required=True, help="run_<timestamp> directory.")
    parser.add_argument(
        "--store",
        type=Path,
        default=None,
        help=f"Output JSONL. Defaults to <run-dir>/{FIRST_TURN_FILE_NAME}.",
    )
    args = parser.parse_args()
    path, exported, skipped = export_run_to_first_turn(args.run_dir, args.store)
    print(f"Exported {exported} first-turn entries to {path}")
    if skipped:
        print(f"Skipped {len(skipped)} samples without a first act: {', '.join(skipped)}")


if __name__ == "__main__":
    main()
