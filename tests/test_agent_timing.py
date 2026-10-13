from __future__ import annotations

import unittest
from types import SimpleNamespace

from smolagents import Tool

from framework.core.agent import _TimedModel, _TimedTool


class AgentTimingTests(unittest.TestCase):
    def test_model_generation_records_duration_and_usage(self):
        class Model:
            @staticmethod
            def generate(*args, **kwargs):
                return SimpleNamespace(
                    token_usage=SimpleNamespace(input_tokens=10, output_tokens=4)
                )

        timings = {"llm_calls": [], "_active_step_number": 2}
        wrapped = _TimedModel(Model(), timings)

        wrapped.generate([])

        self.assertEqual(len(timings["llm_calls"]), 1)
        self.assertEqual(
            timings["llm_calls"][0]["step"],
            2,
        )
        self.assertEqual(timings["llm_calls"][0]["input_tokens"], 10)
        self.assertEqual(timings["llm_calls"][0]["output_tokens"], 4)
        self.assertGreaterEqual(timings["llm_calls"][0]["duration_seconds"], 0)

    def test_tool_call_records_name_step_and_duration(self):
        class EchoTool(Tool):
            name = "echo"
            description = "Return the supplied value."
            inputs = {"value": {"type": "string", "description": "Value."}}
            output_type = "string"

            def forward(self, value: str) -> str:
                return value

        timings = {"tool_calls": [], "_active_step_number": 3}
        wrapped = _TimedTool(EchoTool(), timings)

        self.assertEqual(wrapped.forward(value="ok"), "ok")

        self.assertEqual(len(timings["tool_calls"]), 1)
        self.assertEqual(timings["tool_calls"][0]["tool"], "echo")
        self.assertEqual(timings["tool_calls"][0]["step"], 3)
        self.assertEqual(timings["tool_calls"][0]["status"], "completed")
        self.assertGreaterEqual(timings["tool_calls"][0]["duration_seconds"], 0)


if __name__ == "__main__":
    unittest.main()
