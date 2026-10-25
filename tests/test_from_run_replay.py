from __future__ import annotations

import argparse
import json
import tempfile
import unittest
from pathlib import Path

from smolagents.models import ChatMessage, MessageRole

from evaluation.common import _copy_saved_trajectory, load_saved_actions
from evaluation.first_turn import build_from_run
from framework.core import Agent, local_executor


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


def _write_run(run_dir: Path, file_stem: str, steps: list[dict]) -> Path:
    log_dir = run_dir / "logs" / file_stem
    log_dir.mkdir(parents=True)
    (log_dir / "trajectory.json").write_text(json.dumps(steps), encoding="utf-8")
    (log_dir / "trajectory.txt").write_text("text", encoding="utf-8")
    (log_dir / "raw.json").write_text("{}", encoding="utf-8")
    return log_dir


class FromRunTests(unittest.TestCase):
    def test_load_saved_actions_returns_acts_in_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            _write_run(
                run_dir,
                "000001_s",
                [
                    {"thought": "a", "act": "x = 1", "observation": "{}"},
                    {"thought": "b", "observation": "{}"},
                    {"thought": "c", "act": "send_message(text='hi', objects=[])"},
                ],
            )

            actions = load_saved_actions(run_dir, "000001_s")

            self.assertEqual(actions, ["x = 1", "send_message(text='hi', objects=[])"])

    def test_replay_without_model_calls_when_codes_succeed(self):
        model = FakeModel([])
        agent = Agent(model=model, executor=local_executor())

        result = agent.replay_actions(
            ["x = 1", "send_message(text='hi', objects=[])"], "task", objects={}
        )

        self.assertEqual(model.calls, 0)
        self.assertEqual(result.message.text, "hi")
        self.assertFalse(agent.fallback_used)

    def test_failing_action_falls_back_to_model_once(self):
        model = FakeModel([_structured("send_message(text='recovered', objects=[])")])
        agent = Agent(model=model, executor=local_executor())

        result = agent.replay_actions(["x = 1", "1 / 0"], "task", objects={})

        self.assertEqual(model.calls, 1)
        self.assertEqual(result.message.text, "recovered")
        self.assertTrue(agent.fallback_used)

    def test_copy_saved_trajectory_is_byte_identical(self):
        with tempfile.TemporaryDirectory() as tmp:
            source_run = Path(tmp) / "old"
            source_log = _write_run(
                source_run, "000002_s", [{"thought": "a", "act": "x = 1"}]
            )
            target_dir = Path(tmp) / "new"
            target_dir.mkdir()

            _copy_saved_trajectory(source_run, "000002_s", target_dir)

            for name in ("trajectory.json", "trajectory.txt", "raw.json"):
                self.assertEqual(
                    (target_dir / name).read_bytes(), (source_log / name).read_bytes()
                )
            self.assertFalse((target_dir / "agent_raw_trajectory.json").exists())

    def test_build_from_run_rejects_first_turn_and_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            (run_dir / "logs").mkdir()

            with self.assertRaises(ValueError):
                build_from_run(
                    argparse.Namespace(
                        from_run=run_dir, first_turn="replay", resume=False
                    )
                )
            with self.assertRaises(ValueError):
                build_from_run(
                    argparse.Namespace(from_run=run_dir, first_turn="off", resume=True)
                )
            self.assertIsNone(
                build_from_run(
                    argparse.Namespace(from_run=None, first_turn="off", resume=False)
                )
            )


if __name__ == "__main__":
    unittest.main()
