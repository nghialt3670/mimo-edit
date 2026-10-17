from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from evaluation.common import _write_raw_agent_trajectory, json_safe
from evaluation.compbench_metrics_debug import (
    _raw_prompt_details,
    _resolve_artifact,
)


class CompBenchMetricsDebugTests(unittest.TestCase):
    def test_raw_prompt_details_renders_each_saved_model_input(self):
        with tempfile.TemporaryDirectory() as directory:
            raw_path = Path(directory) / "raw.json"
            raw_path.write_text(
                json.dumps(
                    {
                        "agent_raw_steps": [
                            {"task": "user observation"},
                            {
                                "step_number": 1,
                                "model_input_messages": [
                                    {"role": "system", "content": "system prompt"},
                                    {
                                        "role": "user",
                                        "content": "replace <scribble_0> with a bridge",
                                    },
                                ],
                            },
                            {
                                "step_number": 2,
                                "model_input_messages": [
                                    {"role": "system", "content": "system prompt"},
                                    {"role": "assistant", "content": "prior action"},
                                    {"role": "user", "content": "execution result"},
                                ],
                            },
                        ]
                    }
                ),
                encoding="utf-8",
            )

            rendered = _raw_prompt_details(raw_path)

        self.assertIn("Raw LLM prompts (2 model call(s))", rendered)
        self.assertIn("agent step 1", rendered)
        self.assertIn("agent step 2", rendered)
        self.assertIn("system prompt", rendered)
        self.assertIn("replace &lt;scribble_0&gt; with a bridge", rendered)
        self.assertIn("prior action", rendered)

    def test_resolves_raw_log_from_sample_output_path(self):
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            sample_id = "sample_00001"
            sample_dir = output_dir / "logs" / sample_id
            sample_dir.mkdir(parents=True)
            raw_path = sample_dir / "agent_raw_trajectory.json"
            raw_path.write_text("[]", encoding="utf-8")
            sample_output = str(
                output_dir / "results" / "images" / f"{sample_id}_output.png"
            )

            resolved = _resolve_artifact(
                {},
                "raw_trajectory_path",
                output_dir,
                sample_output,
            )

        self.assertEqual(resolved, raw_path)

    def test_raw_agent_trajectory_is_saved_without_cleaning_prompt_messages(self):
        raw_steps = [
            {
                "step_number": 1,
                "model_input_messages": [
                    {
                        "role": "system",
                        "content": [{"type": "text", "text": "full system prompt"}],
                    },
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": "full observation"}],
                    },
                ],
                "model_output": "raw model output",
                "code_action": "send_message(text='done')",
            }
        ]

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw_path = root / "agent_raw_trajectory.json"
            _write_raw_agent_trajectory(raw_path, raw_steps, root / "assets")
            saved_steps = json.loads(raw_path.read_text(encoding="utf-8"))

        self.assertEqual(saved_steps, raw_steps)

    def test_raw_agent_trajectory_handles_missing_gdbm_while_serializing(self):
        class GdbmDependent:
            def __getattribute__(self, name):
                if name == "__dict__":
                    raise ModuleNotFoundError("No module named '_gdbm'", name="_gdbm")
                return super().__getattribute__(name)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw_path = root / "agent_raw_trajectory.json"
            _write_raw_agent_trajectory(
                raw_path,
                [{"debug_value": GdbmDependent()}],
                root / "assets",
            )
            saved_steps = json.loads(raw_path.read_text(encoding="utf-8"))

        self.assertEqual(
            saved_steps[0]["debug_value"],
            {
                "__type__": "GdbmDependent",
                "__repr__": "Object attributes unavailable because _gdbm is missing",
            },
        )

    def test_json_safe_does_not_suppress_other_missing_modules(self):
        class MissingModule:
            def __getattribute__(self, name):
                if name == "__dict__":
                    raise ModuleNotFoundError(
                        "No module named 'other_module'", name="other_module"
                    )
                return super().__getattribute__(name)

        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ModuleNotFoundError, "other_module"):
                json_safe(MissingModule(), Path(directory))


if __name__ == "__main__":
    unittest.main()
