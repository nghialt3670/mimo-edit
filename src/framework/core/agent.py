from __future__ import annotations

import inspect
import json
import time
from collections.abc import Iterable
from inspect import Parameter, Signature
from types import MethodType
from typing import Any

from smolagents import CodeAgent, Tool
from smolagents.memory import ActionStep, TaskStep

from framework.utils.randomness import seed_random_generators

from . import prompts as _prompts
from .executor import TerminalExecutor, local_executor
from .schema import Message, TurnResult
from .trajectory import clean_trajectory, format_trajectory


def _prompt_text_characters(messages: list[Any]) -> int:
    character_count = 0
    for message in messages:
        content = (
            message.get("content")
            if isinstance(message, dict)
            else getattr(message, "content", None)
        )
        if isinstance(content, str):
            character_count += len(content)
        elif isinstance(content, list):
            for item in content:
                text = (
                    item.get("text")
                    if isinstance(item, dict)
                    else getattr(item, "text", None)
                )
                if isinstance(text, str):
                    character_count += len(text)
    return character_count


class SendMessageTool(Tool):
    name = "send_message"
    description = "Send text and optional Python objects to the user, ending only the current turn."
    inputs = {
        "text": {"type": "string", "description": "Text to send to the user."},
        "objects": {
            "type": "array",
            "description": "Actual Python objects to attach, e.g. [image_2].",
            "nullable": True,
        },
    }
    output_type = "object"

    def forward(self, text: str, objects: list[Any] | None = None) -> dict[str, Any]:
        return {"text": text, "objects": list(objects or [])}


class _ObservationCodeAgent(CodeAgent):
    """Render every task as an Observation, including later user turns."""

    def write_memory_to_messages(self, summary_mode: bool = False):
        started_at = time.perf_counter()
        messages = self.memory.system_prompt.to_messages(summary_mode=summary_mode)
        for step in self.memory.steps:
            step_messages = step.to_messages(summary_mode=summary_mode)
            if isinstance(step, TaskStep) and step_messages:
                step_messages[0].content[0]["text"] = f"Observation:\n{step.task}"
            messages.extend(step_messages)
        timing_records = getattr(self, "_timing_records", None)
        if timing_records is not None and getattr(self, "_active_step_number", None):
            timing_records["prompt_builds"].append(
                {
                    "step": self._active_step_number,
                    "duration_seconds": time.perf_counter() - started_at,
                    "message_count": len(messages),
                    "text_characters": _prompt_text_characters(messages),
                }
            )
        return messages

    def _step_stream(self, memory_step):
        previous_step = getattr(self, "_active_step_number", None)
        self._active_step_number = getattr(memory_step, "step_number", None)
        timing_records = getattr(self, "_timing_records", None)
        if timing_records is not None:
            timing_records["_active_step_number"] = self._active_step_number
        try:
            yield from super()._step_stream(memory_step)
        finally:
            self._active_step_number = previous_step
            if timing_records is not None:
                timing_records["_active_step_number"] = previous_step


class _TimedModel:
    """Record elapsed time and token usage for model generation requests."""

    def __init__(self, model: Any, timing_records: dict[str, list[dict[str, Any]]]):
        self._model = model
        self._timing_records = timing_records

    def __getattr__(self, name: str) -> Any:
        return getattr(self._model, name)

    def generate(self, *args: Any, **kwargs: Any) -> Any:
        started_at = time.perf_counter()
        record: dict[str, Any] = {
            "step": self._timing_records.get("_active_step_number"),
        }
        try:
            response = self._model.generate(*args, **kwargs)
            usage = getattr(response, "token_usage", None)
            if usage is not None:
                if isinstance(usage, dict):
                    record["input_tokens"] = usage.get("input_tokens")
                    record["output_tokens"] = usage.get("output_tokens")
                else:
                    record["input_tokens"] = getattr(usage, "input_tokens", None)
                    record["output_tokens"] = getattr(usage, "output_tokens", None)
            record["status"] = "completed"
            return response
        except Exception as exc:
            record.update({"status": "failed", "error": repr(exc)})
            raise
        finally:
            record["duration_seconds"] = time.perf_counter() - started_at
            self._timing_records["llm_calls"].append(record)

    def generate_stream(self, *args: Any, **kwargs: Any):
        started_at = time.perf_counter()
        record: dict[str, Any] = {
            "step": self._timing_records.get("_active_step_number"),
        }
        try:
            yield from self._model.generate_stream(*args, **kwargs)
            record["status"] = "completed"
        except Exception as exc:
            record.update({"status": "failed", "error": repr(exc)})
            raise
        finally:
            record["duration_seconds"] = time.perf_counter() - started_at
            self._timing_records["llm_calls"].append(record)


class _TimedTool(Tool):
    """Wrap a tool while preserving its public schema and recording call duration."""

    def __init__(
        self, tool: Tool, timing_records: dict[str, list[dict[str, Any]]]
    ) -> None:
        self._tool = tool
        self._timing_records = timing_records
        self.name = tool.name
        self.description = tool.description
        self.inputs = tool.inputs
        self.output_type = tool.output_type

        def timed_forward(_self: _TimedTool, *args: Any, **kwargs: Any) -> Any:
            started_at = time.perf_counter()
            record: dict[str, Any] = {
                "tool": _self.name,
                "step": _self._timing_records.get("_active_step_number"),
            }
            try:
                result = _self._tool.forward(*args, **kwargs)
                record["status"] = "completed"
                return result
            except Exception as exc:
                record.update({"status": "failed", "error": repr(exc)})
                raise
            finally:
                record["duration_seconds"] = time.perf_counter() - started_at
                _self._timing_records["tool_calls"].append(record)

        original_signature = inspect.signature(tool.forward)
        timed_forward.__signature__ = Signature(
            [
                Parameter("self", Parameter.POSITIONAL_OR_KEYWORD),
                *original_signature.parameters.values(),
            ],
            return_annotation=original_signature.return_annotation,
        )
        timed_forward.__annotations__ = getattr(tool.forward, "__annotations__", {})
        self.forward = MethodType(timed_forward, self)
        super().__init__()


class _ObservationFormatter:
    """Normalize execution results to JSON and suppress them after send_message."""

    def __call__(self, step: ActionStep, agent: CodeAgent | None = None) -> None:
        if getattr(step, "is_final_answer", False):
            step.observations = None
            return

        state = (
            getattr(getattr(agent, "python_executor", None), "state", {})
            if agent
            else {}
        )
        names = sorted(
            name for name in state if not name.startswith("_") and name != "__name__"
        )
        payload: dict[str, Any] = {
            "type": "execution_error" if step.error is not None else "execution_result",
            "text": step.observations or "",
            "objects": names,
        }
        if step.error is not None:
            payload["error"] = str(step.error)
        step.observations = json.dumps(payload, ensure_ascii=False)


class Agent:
    """Persistent multi-turn Observation -> Thought -> Python Act conversation.

    One ``Agent`` instance is one conversation. Repeated ``send`` calls append new user
    Observations while preserving both smolagents memory and the Python execution namespace.
    """

    def __init__(
        self,
        *,
        model: Any,
        tools: Iterable[Tool] = (),
        max_steps: int = 12,
        instructions: str | None = None,
        examples: str | None = None,
        executor: Any | None = None,
        verbosity_level: int = 1,
        structured_thought: bool = True,
    ) -> None:
        seed_random_generators()
        self.model = model
        self.tools = list(tools)
        self.max_steps = max_steps
        instruction_chunks = [
            value.strip()
            for value in (instructions, examples)
            if value and value.strip()
        ]
        self.instructions = "\n\n".join(instruction_chunks) or None
        self.executor = executor
        self.verbosity_level = verbosity_level
        self.structured_thought = structured_thought
        self.timing_records: dict[str, Any] = {
            "prompt_builds": [],
            "llm_calls": [],
            "tool_calls": [],
        }
        self._agent = self._build_agent()
        self._started = False

    def _build_agent(self) -> _ObservationCodeAgent:
        base_executor = self.executor or local_executor()
        terminal_executor = (
            base_executor
            if isinstance(base_executor, TerminalExecutor)
            else TerminalExecutor(base_executor)
        )
        # Important: do not assign agent.system_prompt. In smolagents >=1.26 it is read-only.
        # final_answer remains internally registered so the executor can use it, but our Jinja
        # system prompt filters it out of the visible tool definitions.
        agent = _ObservationCodeAgent(
            model=_TimedModel(self.model, self.timing_records),
            tools=[
                _TimedTool(tool, self.timing_records)
                for tool in [*self.tools, SendMessageTool()]
            ],
            prompt_templates=_prompts._PROMPTS,
            instructions=self.instructions,
            executor=terminal_executor,
            max_steps=self.max_steps,
            planning_interval=None,
            return_full_result=True,
            verbosity_level=self.verbosity_level,
            step_callbacks={ActionStep: _ObservationFormatter()},
            use_structured_outputs_internally=self.structured_thought,
        )
        agent._timing_records = self.timing_records
        return agent

    def send(self, text: str, *, objects: dict[str, Any] | None = None) -> TurnResult:
        """Append one user message and run until ``send_message`` terminates this turn."""

        objects = dict(objects or {})
        self._agent.state.update(objects)
        observation = json.dumps(
            {"type": "user_message", "text": text, "objects": list(objects)},
            ensure_ascii=False,
        )
        raw = self._agent.run(
            observation,
            reset=not self._started,
            return_full_result=True,
        )
        self._started = True

        output = raw.output
        if isinstance(output, dict) and "text" in output:
            message = Message(
                text=str(output["text"]), objects=list(output.get("objects") or [])
            )
        else:
            message = Message(text=str(output), objects=[])
        return TurnResult(message=message, raw=raw)

    def reset(self) -> None:
        """Start a fresh conversation and fresh Python namespace."""
        self._agent = self._build_agent()
        self._started = False

    @property
    def variables(self) -> dict[str, Any]:
        state = getattr(self._agent.python_executor, "state", {})
        return {
            key: value
            for key, value in state.items()
            if not key.startswith("_") and key != "__name__"
        }

    @property
    def trajectory(self) -> list[dict[str, Any]]:
        return clean_trajectory(self._agent.memory.get_full_steps())

    @property
    def trajectory_text(self) -> str:
        return format_trajectory(self._agent.memory.get_full_steps())

    @property
    def raw_steps(self) -> list[dict[str, Any]]:
        return list(self._agent.memory.get_full_steps())

    @property
    def timings(self) -> dict[str, list[dict[str, Any]]]:
        return {
            key: [dict(record) for record in self.timing_records[key]]
            for key in ("prompt_builds", "llm_calls", "tool_calls")
        }

    @property
    def inner(self) -> _ObservationCodeAgent:
        """Underlying smolagents agent for advanced use."""
        return self._agent
