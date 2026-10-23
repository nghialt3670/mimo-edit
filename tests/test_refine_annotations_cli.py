from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "src" / "scripts" / "refine_annotations.py"


class RefineAnnotationsCliTests(unittest.TestCase):
    def test_help_as_module(self):
        result = subprocess.run(
            [sys.executable, "-m", "scripts.refine_annotations", "--help"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("SAM 3", result.stdout)

    def test_help_as_script(self):
        result = subprocess.run(
            [sys.executable, str(SCRIPT), "--help"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("SAM 3", result.stdout)

    def test_annotation_directory_resolves_mixed_jsonl(self):
        from scripts.refine_annotations import resolve_annotations_path

        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "run_1"
            run_dir.mkdir()
            jsonl = run_dir / "refedit_bench_mixed.jsonl"
            jsonl.write_text("", encoding="utf-8")
            self.assertEqual(
                resolve_annotations_path(run_dir, default=jsonl), jsonl.resolve()
            )


if __name__ == "__main__":
    unittest.main()
