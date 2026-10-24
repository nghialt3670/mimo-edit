from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from smolagents.models import ChatMessage, MessageRole

from evaluation.first_turn import FirstTurnEntry, FirstTurnStore, task_fingerprint
from framework.core import Agent, local_executor
from framework.core.agent import FirstTurnCommand, first_turn_violation


class FakeModel:
    model_id = "fake"

    def __init__(self, outputs: list[str]) -> None:
        self.outputs = list(outputs)
        self.calls = 0

    def generate(self, messages, stop_sequences=None, **kwargs):
        self.calls += 1
        return ChatMessage(role=MessageRole.ASSISTANT, content=self.outputs.pop(0))


def _structured(code: str, thought: str = "t") -> str:
    return json.dumps({"thought": thought, "code": code})


class FirstTurnTests(unittest.TestCase):
    def test_validator_requires_send_message_last(self):
        self.assertIsNone(first_turn_violation("x = 1\nsend_message(text='hi', objects=[])"))
        self.assertIsNotNone(first_turn_violation("send_message(text='hi')\nx = 1"))
        self.assertIsNotNone(first_turn_violation("x = 1"))
        self.assertIsNotNone(first_turn_violation("def ("))

    def test_generate_does_not_execute_and_reprompts_on_invalid_ending(self):
        model = FakeModel(
            [
                _structured("x = 1"),
                _structured("send_message(text='hi', objects=[])"),
            ]
        )
        agent = Agent(model=model, executor=local_executor())

        command = agent.generate_first_turn("task", objects={})

        self.assertEqual(model.calls, 2)
        self.assertEqual(command.code, "send_message(text='hi', objects=[])")
        self.assertEqual(command.thought, "t")
        self.assertNotIn("x", agent.variables)

    def test_replay_success_finishes_without_model_calls(self):
        model = FakeModel([])
        agent = Agent(model=model, executor=local_executor())
        command = FirstTurnCommand(
            code="send_message(text='hi', objects=[])",
            thought="t",
            raw_output=_structured("send_message(text='hi', objects=[])"),
        )

        result = agent.replay_first_turn(command, "task", objects={})

        self.assertEqual(model.calls, 0)
        self.assertEqual(result.message.text, "hi")

    def test_replay_error_continues_react_loop(self):
        model = FakeModel([_structured("send_message(text='recovered', objects=[])")])
        agent = Agent(model=model, executor=local_executor())
        command = FirstTurnCommand(
            code="1 / 0\nsend_message(text='x', objects=[])",
            thought="t",
            raw_output=_structured("1 / 0\nsend_message(text='x', objects=[])"),
        )

        result = agent.replay_first_turn(command, "task", objects={})

        self.assertEqual(model.calls, 1)
        self.assertEqual(result.message.text, "recovered")

    def test_store_round_trip_last_entry_wins(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "nested" / "first_turn.jsonl"
            fingerprint = task_fingerprint("task", {"image_1": object()})
            entry = FirstTurnEntry(
                sample_id="s1",
                prompt_mode="mixed",
                prompt_signature="box",
                fingerprint=fingerprint,
                task="task",
                objects=["image_1"],
                thought="t",
                code="send_message(text='a', objects=[])",
                raw_output="raw",
                model_id="fake",
            )
            store = FirstTurnStore(path)
            store.put(entry)
            store.put(replace(entry, code="send_message(text='b', objects=[])"))

            reloaded = FirstTurnStore(path).get("s1", "mixed", "box")

            self.assertIsNotNone(reloaded)
            self.assertEqual(reloaded.code, "send_message(text='b', objects=[])")
            self.assertEqual(reloaded.fingerprint, fingerprint)
            self.assertIsNone(FirstTurnStore(path).get("s1", "text", "box"))
            self.assertIsNone(FirstTurnStore(path).get("s1", "mixed", "point"))

    def test_store_keeps_one_entry_per_prompt_signature(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "first_turn.jsonl"
            base = FirstTurnEntry(
                sample_id="s1",
                prompt_mode="mixed",
                prompt_signature="box",
                fingerprint="f",
                task="Remove <box_0>.",
                objects=["box_0", "image_0"],
                thought="t",
                code="box code",
                raw_output="raw",
                model_id="fake",
            )
            store = FirstTurnStore(path)
            store.put(base)
            store.put(
                replace(
                    base,
                    prompt_signature="point",
                    task="Remove <point_0>.",
                    code="point code",
                )
            )

            reloaded = FirstTurnStore(path)

            self.assertEqual(reloaded.get("s1", "mixed", "box").code, "box code")
            self.assertEqual(reloaded.get("s1", "mixed", "point").code, "point code")

    def test_store_ignores_legacy_lines_without_prompt_signature(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "first_turn.jsonl"
            legacy = {
                "sample_id": "s1",
                "prompt_mode": "mixed",
                "fingerprint": "f",
                "task": "Remove <box_0>.",
                "objects": ["box_0", "image_0"],
                "thought": "t",
                "code": "legacy",
                "raw_output": "raw",
                "model_id": None,
            }
            path.write_text(json.dumps(legacy) + "\n", encoding="utf-8")

            self.assertIsNone(FirstTurnStore(path).get("s1", "mixed", "box"))


if __name__ == "__main__":
    unittest.main()
