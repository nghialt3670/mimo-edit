from __future__ import annotations

import ast
import inspect
import json
import time
from collections.abc import Iterable
from dataclasses import dataclass
from inspect import Parameter, Signature
from types import MethodType
from typing import Any

from smolagents import CodeAgent, Tool
from smolagents.memory import ActionStep, SystemPromptStep, TaskStep, Timing, ToolCall
from smolagents.models import CODEAGENT_RESPONSE_FORMAT, ChatMessage, MessageRole
from smolagents.utils import (
    AgentExecutionError,
    extract_code_from_text,
    parse_code_blobs,
    truncate_content,
)

from framework.utils.randomness import seed_random_generators

from . import prompts as _prompts
from .executor import TerminalExecutor, local_executor
from .schema import Message, TurnResult
from .trajectory import clean_trajectory, format_trajectory

FIRST_TURN_INSTRUCTION = (
    "This is the first turn. Write one complete code action that handles the whole task "
    "and ends with a send_message(text=..., objects=[...]) call."
)


@dataclass(frozen=True, slots=True)
class FirstTurnCommand:
    """The first code action for a turn, generated without executing it."""

    code: str
    thought: str
    raw_output: str


def first_turn_violation(code: str) -> str | None:
    """Return why ``code`` cannot end a turn, or None when it ends with send_message."""
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return f"the code does not parse ({exc})."
    if tree.body and isinstance(tree.body[-1], ast.Expr):
        call = tree.body[-1].value
        if (
            isinstance(call, ast.Call)
            and isinstance(call.func, ast.Name)
            and call.func.id == "send_message"
        ):
            return None
    return "the code must end with a send_message(...) call."


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

    @staticmethod
    def _user_observation(text: str, objects: dict[str, Any]) -> str:
        return json.dumps(
            {"type": "user_message", "text": text, "objects": list(objects)},
            ensure_ascii=False,
        )

    @staticmethod
    def _turn_result(output: Any, raw: Any) -> TurnResult:
        if isinstance(output, dict) and "text" in output:
            message = Message(
                text=str(output["text"]), objects=list(output.get("objects") or [])
            )
        else:
            message = Message(text=str(output), objects=[])
        return TurnResult(message=message, raw=raw)

    def send(self, text: str, *, objects: dict[str, Any] | None = None) -> TurnResult:
        """Append one user message and run until ``send_message`` terminates this turn."""

        objects = dict(objects or {})
        self._agent.state.update(objects)
        raw = self._agent.run(
            self._user_observation(text, objects),
            reset=not self._started,
            return_full_result=True,
        )
        self._started = True
        return self._turn_result(raw.output, raw)

    def _begin_turn(self, text: str, objects: dict[str, Any] | None) -> None:
        """Reset memory to a single first user message, as ``run`` does for a new task."""
        objects = dict(objects or {})
        inner = self._agent
        inner.state.update(objects)
        observation = self._user_observation(text, objects)
        inner.task = observation
        inner.interrupt_switch = False  # normally set by smolagents' run()
        inner.memory.system_prompt = SystemPromptStep(system_prompt=inner.system_prompt)
        inner.memory.reset()
        inner.monitor.reset()
        inner.memory.steps.append(TaskStep(task=observation))
        inner.python_executor.send_variables(variables=inner.state)
        inner.python_executor.send_tools({**inner.tools, **inner.managed_agents})
        self._started = True

    def _parse_first_turn_output(self, raw: str) -> FirstTurnCommand:
        tags = self._agent.code_block_tags
        if self.structured_thought:
            payload = json.loads(raw)
            thought = str(payload.get("thought") or "").strip()
            code = str(payload["code"])
            code = extract_code_from_text(code, tags) or code
        else:
            thought = ""
            code = parse_code_blobs(raw, tags)
        return FirstTurnCommand(code=code.strip(), thought=thought, raw_output=raw)

    def first_turn_messages(
        self, text: str, *, objects: dict[str, Any] | None = None
    ) -> list[ChatMessage]:
        """Return the model messages for the first turn without calling the model."""
        self._begin_turn(text, objects)
        messages = self._agent.write_memory_to_messages()
        messages.append(
            ChatMessage(
                role=MessageRole.USER,
                content=[{"type": "text", "text": FIRST_TURN_INSTRUCTION}],
            )
        )
        return messages

    def first_turn_request_options(self) -> dict[str, Any]:
        """Return the ``generate`` keyword arguments for first-turn requests."""
        options: dict[str, Any] = {"stop_sequences": ["Observation:", "Calling tools:"]}
        if self.structured_thought:
            options["response_format"] = CODEAGENT_RESPONSE_FORMAT
        return options

    def parse_first_turn_output(self, raw: str) -> FirstTurnCommand:
        """Parse one raw model output into a first-turn command."""
        return self._parse_first_turn_output(raw)

    def generate_first_turn(
        self,
        text: str,
        *,
        objects: dict[str, Any] | None = None,
        max_attempts: int = 3,
    ) -> FirstTurnCommand:
        """Ask the model for one first-turn code action without executing it.

        The model is re-prompted with the rejection reason when the output cannot be
        parsed or does not end with ``send_message``.
        """
        messages = self.first_turn_messages(text, objects=objects)
        request_options = self.first_turn_request_options()
        last_raw = ""
        problem = ""
        for _ in range(max_attempts):
            if problem:
                messages.append(
                    ChatMessage(
                        role=MessageRole.ASSISTANT,
                        content=[{"type": "text", "text": last_raw}],
                    )
                )
                messages.append(
                    ChatMessage(
                        role=MessageRole.USER,
                        content=[
                            {
                                "type": "text",
                                "text": f"Rejected: {problem} Respond again with one complete first-turn code action.",
                            }
                        ],
                    )
                )
            chat = self.model.generate(messages, **request_options)
            last_raw = chat.content or ""
            try:
                command = self._parse_first_turn_output(last_raw)
            except Exception as exc:
                problem = f"the output could not be parsed ({exc})."
                continue
            problem = first_turn_violation(command.code) or ""
            if not problem:
                return command
        raise RuntimeError(f"First-turn generation failed: {problem}")

    def replay_first_turn(
        self,
        command: FirstTurnCommand,
        text: str,
        *,
        objects: dict[str, Any] | None = None,
    ) -> TurnResult:
        """Execute a saved first-turn action.

        The ReAct loop continues with the model only when the execution raises an error
        or does not finish the turn.
        """
        self._begin_turn(text, objects)
        inner = self._agent
        step = ActionStep(
            step_number=1,
            timing=Timing(start_time=time.time()),
            model_output_message=ChatMessage(
                role=MessageRole.ASSISTANT, content=command.raw_output
            ),
            model_output=command.raw_output,
            code_action=command.code,
            tool_calls=[
                ToolCall(name="python_interpreter", arguments=command.code, id="call_1")
            ],
        )
        inner.memory.steps.append(step)
        code_output = None
        try:
            code_output = inner.python_executor(command.code)
            step.observations = (
                "Execution logs:\n"
                + code_output.logs
                + "Last output from code snippet:\n"
                + truncate_content(str(code_output.output))
            )
            step.action_output = code_output.output
            step.is_final_answer = code_output.is_final_answer
        except Exception as exc:
            step.error = AgentExecutionError(str(exc), inner.logger)
        step.timing.end_time = time.time()
        _ObservationFormatter()(step, agent=inner)

        if step.error is None and step.is_final_answer:
            return self._turn_result(code_output.output, raw=None)
        steps = list(inner._run_stream(task=inner.task, max_steps=inner.max_steps))
        return self._turn_result(steps[-1].output, raw=None)

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
