import unittest

from tools.registry import ToolRegistry


class ToolRegistryDispatchTests(unittest.TestCase):
    def test_registered_tool_accepts_positional_and_keyword_arguments(self):
        registry = ToolRegistry()

        @registry.register(
            "combine",
            "Combine two values.",
            {
                "first": {"type": "string", "description": "First value."},
                "second": {
                    "type": "string",
                    "description": "Second value.",
                    "nullable": True,
                },
            },
            "string",
        )
        def combine(context, first, second):
            return f"{context}:{first}:{second}"

        tool = registry.build("context")[0]
        self.assertEqual(tool.forward("one", second="two"), "context:one:two")
        self.assertEqual(tool.forward(first="one"), "context:one:None")

    def test_registered_tool_rejects_duplicate_positional_and_keyword_argument(self):
        registry = ToolRegistry()

        @registry.register(
            "echo",
            "Echo a value.",
            {"value": {"type": "string", "description": "Value."}},
            "string",
        )
        def echo(context, value):
            return value

        tool = registry.build()[0]
        with self.assertRaisesRegex(TypeError, "multiple values"):
            tool.forward("positional", value="keyword")
