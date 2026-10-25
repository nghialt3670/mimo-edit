import unittest

from evaluation.common import build_agent_context
from framework.tools import image_editing_tools

FORBIDDEN_TEXT_WORDS = ("box", "point", "scribble", "selector", "mask", "crop")


def _tool(tools, name):
    matches = [tool for tool in tools if tool.name == name]
    assert len(matches) == 1, name
    return matches[0]


class TextModeToolTests(unittest.TestCase):
    def test_text_mode_segment_object_takes_only_text(self):
        tools = image_editing_tools(pool=object(), prompt_mode="text")
        segment = _tool(tools, "segment_object")
        self.assertEqual(set(segment.inputs), {"image", "text"})

    def test_mixed_mode_segment_object_keeps_selectors(self):
        tools = image_editing_tools(pool=object(), prompt_mode="mixed")
        segment = _tool(tools, "segment_object")
        self.assertTrue({"point", "box", "scribble"} <= set(segment.inputs))

    def test_unknown_prompt_mode_is_rejected(self):
        with self.assertRaises(ValueError):
            image_editing_tools(pool=object(), prompt_mode="bogus")


class TextModeInstructionTests(unittest.TestCase):
    def _sample(self):
        return {
            "instruction": "Make the cat red.",
            "references": [{"token": "<ref_0>", "span": "the cat"}],
        }

    def test_text_instructions_have_no_selector_wording(self):
        text = build_agent_context(self._sample(), "text").lower()
        for word in FORBIDDEN_TEXT_WORDS:
            self.assertNotIn(word, text)
        self.assertIn("segment_object(text=...)", text)

    def test_mixed_instructions_keep_selector_guidance(self):
        text = build_agent_context(self._sample(), "mixed")
        self.assertIn("segment_object", text)
        self.assertIn("Do not invent selector coordinates.", text)


if __name__ == "__main__":
    unittest.main()
