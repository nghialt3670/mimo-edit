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
