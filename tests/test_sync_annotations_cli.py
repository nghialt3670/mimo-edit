from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "src" / "scripts" / "sync_annotations.py"


class SyncAnnotationsCliTests(unittest.TestCase):
    def test_direct_script_invocation(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            annotations = Path(temporary_directory) / "annotations.jsonl"
            audit = Path(temporary_directory) / "audit.jsonl"
            annotations.write_text("", encoding="utf-8")
            audit.write_text("", encoding="utf-8")

            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "refedit_bench",
                    "--annotations",
                    str(annotations),
                    "--audit",
                    str(audit),
                ],
                cwd=ROOT,
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("already sorted", result.stdout)

    def test_module_invocation(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            annotations = Path(temporary_directory) / "annotations.jsonl"
            audit = Path(temporary_directory) / "audit.jsonl"
            annotations.write_text("", encoding="utf-8")
            audit.write_text("", encoding="utf-8")

            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "scripts.sync_annotations",
                    "refedit_bench",
                    "--annotations",
                    str(annotations),
                    "--audit",
                    str(audit),
                ],
                cwd=ROOT,
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("already sorted", result.stdout)

    def test_mira_bench_benchmark(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            annotations = Path(temporary_directory) / "annotations.jsonl"
            audit = Path(temporary_directory) / "audit.jsonl"
            annotations.write_text("", encoding="utf-8")
            audit.write_text("", encoding="utf-8")

            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "mira_bench",
                    "--annotations",
                    str(annotations),
                    "--audit",
                    str(audit),
                ],
                cwd=ROOT,
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("already sorted", result.stdout)


if __name__ == "__main__":
    unittest.main()
