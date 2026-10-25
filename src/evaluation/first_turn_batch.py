"""Generate first-turn commands through the Gemini Batch API.

Workflow: ``build_requests`` -> ``write_batch_inputs`` -> ``submit_batch`` (runs
submit_batch.py and waits) -> ``collect_results`` on batch_output.jsonl. Results are
written into a ``FirstTurnStore`` so that inference can replay them with ``--first-turn replay``.
"""

from __future__ import annotations

import json
import subprocess
import sys
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from framework.core import Agent, local_executor
from framework.core.agent import first_turn_violation
from framework.utils.randomness import seed_random_generators

from .common import (
    AGENT_EXECUTION_TIMEOUT_SECONDS,
    build_agent_context,
    build_agent_task,
    load_sample,
    select_reference_prompts,
)
from .first_turn import (
    FirstTurnEntry,
    FirstTurnStore,
    prompt_signature,
    task_fingerprint,
)

BATCH_ENDPOINT = "/v1/chat/completions"
REQUESTS_FILE = "requests.jsonl"
OUTPUT_FILE = "batch_output.jsonl"
MANIFEST_FILE = "manifest.json"
SUBMIT_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "submit_batch.py"


@dataclass(frozen=True, slots=True)
class FirstTurnRequest:
    custom_id: str
    sample_id: str
    prompt_mode: str
    prompt_signature: str
    fingerprint: str
    task: str
    objects: list[str]
    body: dict[str, Any]


@dataclass(slots=True)
class PrepareReport:
    requests: list[FirstTurnRequest] = field(default_factory=list)
    cached: int = 0
    errors: list[tuple[str, str]] = field(default_factory=list)


@dataclass(slots=True)
class CollectReport:
    stored: int = 0
    missing: list[str] = field(default_factory=list)
    failed: list[tuple[str, str]] = field(default_factory=list)


def custom_id_for(annotation_index: int) -> str:
    """Return the batch request id; it maps back to the annotation index."""
    return f"{annotation_index:06d}"


def build_requests(
    records: Sequence[dict[str, Any]],
    sample_indices: Iterable[int],
    *,
    benchmark: str,
    prompt_mode: str,
    image_loader: Any,
    model: Any,
    tools: list[Any],
    store: FirstTurnStore,
) -> PrepareReport:
    """Build batch requests for samples that are not yet cached with a matching fingerprint."""
    report = PrepareReport()
    for annotation_index in sample_indices:
        try:
            request = _build_request(
                records[annotation_index],
                annotation_index,
                benchmark=benchmark,
                prompt_mode=prompt_mode,
                image_loader=image_loader,
                model=model,
                tools=tools,
            )
        except Exception as exc:  # one bad sample must not stop the whole batch
            report.errors.append(
                (custom_id_for(annotation_index), f"{type(exc).__name__}: {exc}")
            )
            continue
        cached = store.get(request.sample_id, prompt_mode, request.prompt_signature)
        if cached is not None and cached.fingerprint == request.fingerprint:
            report.cached += 1
            continue
        report.requests.append(request)
    return report


def _build_request(
    record: dict[str, Any],
    annotation_index: int,
    *,
    benchmark: str,
    prompt_mode: str,
    image_loader: Any,
    model: Any,
    tools: list[Any],
) -> FirstTurnRequest:
    # Mirrors the preparation in evaluation.common.run_one so the prompt is identical.
    seed_random_generators()
    sample_id = record.get("sample_id") or f"annotation_{annotation_index}"
    sample = load_sample(
        record, benchmark=benchmark, image_loader=image_loader, prompt_mode=prompt_mode
    )
    sample["sample_id"] = sample.get("sample_id") or sample_id
    select_reference_prompts(sample, prompt_mode)
    task, objects = build_agent_task(sample)
    signature = prompt_signature(
        ref["selected_prompt"]["type"]
        for ref in sample["references"]
        if ref.get("selected_prompt") is not None
    )
    agent = Agent(
        model=model,
        tools=tools,
        instructions=build_agent_context(sample, prompt_mode),
        executor=local_executor(timeout_seconds=AGENT_EXECUTION_TIMEOUT_SECONDS),
        verbosity_level=0,
    )
    messages = agent.first_turn_messages(task, objects=objects)
    body = model._prepare_completion_kwargs(
        messages=messages,
        model=model.model_id,
        custom_role_conversions=model.custom_role_conversions,
        convert_images_to_image_urls=True,
        **agent.first_turn_request_options(),
    )
    body.setdefault("model", model.model_id)
    return FirstTurnRequest(
        custom_id=custom_id_for(annotation_index),
        sample_id=sample["sample_id"],
        prompt_mode=prompt_mode,
        prompt_signature=signature,
        fingerprint=task_fingerprint(task, objects),
        task=task,
        objects=sorted(objects),
        body=body,
    )


def write_batch_inputs(
    work_dir: Path,
    requests: Sequence[FirstTurnRequest],
    *,
    benchmark: str,
    model_id: str,
    store_path: Path,
) -> None:
    """Write the JSONL batch input and a manifest used to map results back to samples."""
    work_dir.mkdir(parents=True, exist_ok=True)
    with (work_dir / REQUESTS_FILE).open("w", encoding="utf-8") as handle:
        for request in requests:
            line = {
                "custom_id": request.custom_id,
                "method": "POST",
                "url": BATCH_ENDPOINT,
                "body": request.body,
            }
            handle.write(json.dumps(line, ensure_ascii=False) + "\n")
    manifest = {
        "benchmark": benchmark,
        "model_id": model_id,
        "store_path": str(store_path),
        "requests": {
            request.custom_id: {
                "sample_id": request.sample_id,
                "prompt_mode": request.prompt_mode,
                "prompt_signature": request.prompt_signature,
                "fingerprint": request.fingerprint,
                "task": request.task,
                "objects": request.objects,
            }
            for request in requests
        },
    }
    (work_dir / MANIFEST_FILE).write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def load_manifest(work_dir: Path) -> dict[str, Any]:
    return json.loads((work_dir / MANIFEST_FILE).read_text(encoding="utf-8"))


def submit_batch(work_dir: Path) -> int:
    """Run submit_batch.py on the work directory's requests and wait for it to finish."""
    work_dir = work_dir.resolve()
    result = subprocess.run(
        [sys.executable, str(SUBMIT_SCRIPT), str(work_dir / REQUESTS_FILE)],
        cwd=work_dir,
        check=False,
    )
    return result.returncode


def parse_batch_output(
    lines: Iterable[str],
    requests: dict[str, dict[str, Any]],
    *,
    store: FirstTurnStore,
    parser: Any,
    model_id: str,
) -> CollectReport:
    """Validate batch output lines and store accepted first-turn commands.

    ``parser`` must provide ``parse_first_turn_output(raw) -> FirstTurnCommand``.
    """
    report = CollectReport()
    seen: set[str] = set()
    for line in lines:
        if not line.strip():
            continue
        record = json.loads(line)
        custom_id = str(record.get("custom_id"))
        seen.add(custom_id)
        meta = requests.get(custom_id)
        if meta is None:
            report.failed.append((custom_id, "unknown custom_id"))
            continue
        response = record.get("response") or {}
        if record.get("error") or response.get("status_code") != 200:
            report.failed.append((custom_id, f"request failed: {record.get('error') or response.get('status_code')}"))
            continue
        try:
            content = response["body"]["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError):
            report.failed.append((custom_id, "malformed response body"))
            continue
        try:
            command: FirstTurnCommand = parser.parse_first_turn_output(content)
        except Exception as exc:
            report.failed.append((custom_id, f"parse error: {type(exc).__name__}"))
            continue
        violation = first_turn_violation(command.code)
        if violation:
            report.failed.append((custom_id, f"invalid output: {violation}"))
            continue
        store.put(
            FirstTurnEntry(
                sample_id=meta["sample_id"],
                prompt_mode=meta["prompt_mode"],
                prompt_signature=meta["prompt_signature"],
                fingerprint=meta["fingerprint"],
                task=meta["task"],
                objects=list(meta["objects"]),
                thought=command.thought,
                code=command.code,
                raw_output=content,
                model_id=model_id,
            )
        )
        report.stored += 1
    report.missing = sorted(set(requests) - seen)
    return report


def collect_results(
    work_dir: Path,
    *,
    parser: Any,
    output_file: Path | None = None,
    store_path: Path | None = None,
) -> CollectReport:
    """Store accepted first-turn commands from a local output file or the completed batch download."""
    manifest = load_manifest(work_dir)
    path = output_file or work_dir / OUTPUT_FILE
    lines = path.read_text(encoding="utf-8").splitlines()
    store = FirstTurnStore(store_path or Path(manifest["store_path"]))
    return parse_batch_output(
        lines,
        manifest["requests"],
        store=store,
        parser=parser,
        model_id=manifest["model_id"],
    )
