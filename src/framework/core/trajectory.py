from __future__ import annotations

import ast
import json
import re
from typing import Any


def clean_trajectory(steps: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Reduce smolagents' verbose memory dump to Observation / Thought / Act records."""

    out: list[dict[str, Any]] = []
    for step in steps:
        if "task" in step:
            out.append({"observation": _parse_task(step.get("task"))})
            continue

        code = step.get("code_action")
        thought = _extract_thought(step.get("model_output"))
        if not thought:
            thought = _extract_thought(step.get("model_output_message"))

        record: dict[str, Any] = {}
        if thought:
            record["thought"] = thought
        if code:
            record["act"] = str(code).strip()

        # send_message is terminal for one turn. The next Observation is the next user message.
        if not step.get("is_final_answer"):
            observation = _parse_observation(
                step.get("observations"), step.get("error")
            )
            if observation is not None:
                record["observation"] = observation

        if record:
            out.append(record)

    return out


def format_trajectory(steps: list[dict[str, Any]]) -> str:
    chunks: list[str] = []
    for item in clean_trajectory(steps):
        if "observation" in item:
            chunks.append("Observation:\n" + _pretty(item["observation"]))
        if "thought" in item:
            chunks.append("Thought:\n" + str(item["thought"]).strip())
        if "act" in item:
            chunks.append("Act:\n" + str(item["act"]).strip())
    return "\n\n".join(chunks)


def _parse_task(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    text = value.strip()
    try:
        return json.loads(text)
    except Exception:
        try:
            return ast.literal_eval(text)
        except Exception:
            return text


def _parse_observation(value: Any, error: Any = None) -> Any | None:
    if value in (None, "") and error is None:
        return None
    if isinstance(value, str):
        try:
            return json.loads(value)
        except Exception:
            pass
    if error is not None:
        return {
            "type": "execution_error",
            "text": str(value or ""),
            "error": str(error),
        }
    return value


def _extract_thought(value: Any) -> str:
    """Extract explicit thought from structured or textual CodeAgent output."""

    if value is None:
        return ""

    if isinstance(value, dict):
        # Whole ChatMessage-like dictionary.
        if "thought" in value or "thoughts" in value:
            return str(value.get("thought") or value.get("thoughts") or "").strip()
        if "content" in value:
            return _extract_thought(value["content"])

    if isinstance(value, list):
        for item in value:
            thought = _extract_thought(item)
            if thought:
                return thought
        return ""

    text = str(value).strip()
    if not text:
        return ""

    # Structured CodeAgent output: {"thought":"...","code":"..."}
    try:
        payload = json.loads(text)
        if isinstance(payload, dict):
            return str(payload.get("thought") or payload.get("thoughts") or "").strip()
    except Exception:
        pass

    # Python-dict-like rendering can occur in debug dumps.
    try:
        payload = ast.literal_eval(text)
        if isinstance(payload, dict):
            return str(payload.get("thought") or payload.get("thoughts") or "").strip()
    except Exception:
        pass

    # Classical textual CodeAgent output.
    match = re.search(
        r"(?:^|\n)\s*Thought\s*:\s*(.*?)(?=\n\s*(?:<code>|```(?:python|py)?|Code\s*:)|$)",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    return match.group(1).strip() if match else ""


def _pretty(value: Any) -> str:
    if isinstance(value, (dict, list)):
        return json.dumps(value, indent=2, ensure_ascii=False, default=str)
    return str(value)
