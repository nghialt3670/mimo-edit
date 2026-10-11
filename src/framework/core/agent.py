from __future__ import annotations

import json
from collections.abc import Iterable
from typing import Any

from smolagents import CodeAgent, Tool
from smolagents.memory import ActionStep, TaskStep

from framework.utils.randomness import seed_random_generators

from .executor import TerminalExecutor, local_executor
from .schema import Message, TurnResult
from .trajectory import clean_trajectory, format_trajectory

_SYSTEM_PROMPT = r"""
You are a code-action agent operating in a repeated Observation -> Thought -> Act loop.

An Observation is JSON. A user observation has this form:
{"type":"user_message","text":"...","objects":["image_0","point_0", ...]}

The `objects` entries are Python variable names already bound in your persistent execution
namespace. Use those variables directly. In mixed visual-prompt mode, placeholders name the
exact supplied variable, such as `<point_0>` for `point_0` or `<box_1>` for `box_1`. Variables
you create in code persist across later steps and later user turns. Never invent a variable that
has not appeared in an Observation or been created by your own earlier code.

At every agent step you MUST produce exactly two structured fields:
- `thought`: one concise plain-text reasoning string describing the next action.
- `code`: valid Python implementing that action.

The framework enforces this structured Thought + Code output. Do not omit `thought`.
Registered tools behave like ordinary Python functions. Assign useful intermediate outputs to
clear names such as `mask_0`, `image_1`, and `image_2`.

When the CURRENT USER TURN is complete, terminate the turn with:
    send_message(text="...", objects=[...])

`send_message` immediately returns that message to the user. It does NOT end the conversation.
The next item in the loop, if the user replies, will be a new user-message Observation. Do not
produce another execution Observation after send_message. Put actual Python objects in `objects`,
not variable names as strings. Never call `final_answer`; it is an internal implementation detail.

Available functions:
```python
{%- for name, tool in tools.items() %}
{%- if name != "final_answer" %}
{{ tool.to_code_prompt() }}
{%- endif %}
{% endfor %}
```

Authorized imports: {{authorized_imports}}

{%- if custom_instructions %}
Additional instructions / few-shot exemplars:
{{custom_instructions}}
{%- endif %}
""".strip()

_PROMPTS = {
    "system_prompt": _SYSTEM_PROMPT,
    "planning": {
        "initial_plan": "{{task}}",
        "update_plan_pre_messages": "{{task}}",
        "update_plan_post_messages": "Continue. Remaining steps: {{remaining_steps}}",
    },
    "managed_agent": {"task": "{{task}}", "report": "{{final_answer}}"},
    "final_answer": {
        "pre_messages": "The turn ended before send_message was called.",
        "post_messages": "Return a concise response for this turn: {{task}}",
    },
}


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
        messages = self.memory.system_prompt.to_messages(summary_mode=summary_mode)
        for step in self.memory.steps:
            step_messages = step.to_messages(summary_mode=summary_mode)
            if isinstance(step, TaskStep) and step_messages:
                step_messages[0].content[0]["text"] = f"Observation:\n{step.task}"
            messages.extend(step_messages)
        return messages


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
        self.instructions = self._join_instructions(instructions, examples)
        self.executor = executor
        self.verbosity_level = verbosity_level
        self.structured_thought = structured_thought
        self._agent = self._build_agent()
        self._started = False
        self._last_raw: Any | None = None

    @staticmethod
    def _join_instructions(
        instructions: str | None, examples: str | None
    ) -> str | None:
        chunks = [x.strip() for x in (instructions, examples) if x and x.strip()]
        return "\n\n".join(chunks) or None

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
        return _ObservationCodeAgent(
            model=self.model,
            tools=[*self.tools, SendMessageTool()],
            prompt_templates=_PROMPTS,
            instructions=self.instructions,
            executor=terminal_executor,
            max_steps=self.max_steps,
            planning_interval=None,
            return_full_result=True,
            verbosity_level=self.verbosity_level,
            step_callbacks={ActionStep: _ObservationFormatter()},
            use_structured_outputs_internally=self.structured_thought,
        )

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
        self._last_raw = raw

        output = raw.output
        if isinstance(output, dict) and "text" in output:
            message = Message(
                text=str(output["text"]), objects=list(output.get("objects") or [])
            )
        else:
            message = Message(text=str(output), objects=[])
        return TurnResult(message=message, raw=raw)

    __call__ = send

    def reset(self) -> None:
        """Start a fresh conversation and fresh Python namespace."""
        self._agent = self._build_agent()
        self._started = False
        self._last_raw = None

    @property
    def variables(self) -> dict[str, Any]:
        state = getattr(self._agent.python_executor, "state", {})
        return {
            key: value
            for key, value in state.items()
            if not key.startswith("_") and key != "__name__"
        }

    @property
    def system_prompt(self) -> str:
        return self._agent.system_prompt

    @property
    def messages(self):
        return self._agent.write_memory_to_messages()

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
    def inner(self) -> _ObservationCodeAgent:
        """Underlying smolagents agent for advanced use."""
        return self._agent

    def replay(self, *, detailed: bool = False) -> None:
        self._agent.replay(detailed=detailed)
