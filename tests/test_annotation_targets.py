from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.annotation_targets import annotation_target


class AnnotationTargetTests(unittest.TestCase):
    def test_mira_bench_uses_latest_run_directory(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_root = Path(temporary_directory) / "mira_bench"
            older_run = output_root / "run_20261014_210000_000000"
            latest_run = output_root / "run_20261015_210158_946635"
            older_run.mkdir(parents=True)
            latest_run.mkdir()

            with (
                patch.dict(os.environ, {"ANNOTATE_OUT": str(output_root)}),
                patch("scripts.annotation_targets.load_project_environment"),
            ):
                target = annotation_target("mira_bench", Path(temporary_directory))

            self.assertEqual(target.annotations, latest_run / "mira_bench_mixed.jsonl")
            self.assertEqual(target.audit, latest_run / "mira_bench_audit.jsonl")

    def test_mira_bench_falls_back_to_legacy_mirage_run(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            legacy_run = (
                Path(temporary_directory)
                / ".out"
                / "annotation"
                / "mirage"
                / "run_20261015_210158_946635"
            )
            legacy_run.mkdir(parents=True)
            (legacy_run / "mirage_mixed.jsonl").write_text("", encoding="utf-8")

            with patch("scripts.annotation_targets.load_project_environment"):
                target = annotation_target("mira_bench", Path(temporary_directory))

            self.assertEqual(target.annotations, legacy_run / "mirage_mixed.jsonl")
            self.assertEqual(target.audit, legacy_run / "mirage_audit.jsonl")


if __name__ == "__main__":
    unittest.main()
