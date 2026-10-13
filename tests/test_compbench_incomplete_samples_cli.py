from __future__ import annotations

import sys
import unittest
from unittest.mock import patch

from scripts.compbench_incomplete_samples import _parse_args


class CompBenchIncompleteSamplesCliTests(unittest.TestCase):
    def test_defaults_to_mixed_mode(self):
        with patch.object(sys, "argv", ["compbench_incomplete_samples"]):
            args = _parse_args()

        self.assertEqual(args.prompt_mode, "mixed")

    def test_accepts_text_mode(self):
        with patch.object(
            sys,
            "argv",
            ["compbench_incomplete_samples", "--prompt-mode", "text"],
        ):
            args = _parse_args()

        self.assertEqual(args.prompt_mode, "text")


if __name__ == "__main__":
    unittest.main()
