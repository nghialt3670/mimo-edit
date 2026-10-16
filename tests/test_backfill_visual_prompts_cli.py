from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class BackfillVisualPromptsCliTests(unittest.TestCase):
    def test_mira_bench_benchmark(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            annotations = Path(temporary_directory) / "annotations.jsonl"
            annotations.write_text("", encoding="utf-8")

            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "scripts.backfill_visual_prompts",
                    "mira_bench",
                    "--annotations",
                    str(annotations),
                ],
                cwd=ROOT,
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("'records': 0", result.stdout)


if __name__ == "__main__":
    unittest.main()
