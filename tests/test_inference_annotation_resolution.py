from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from evaluation.notebook import resolve_annotation_file


class AnnotationFileResolutionTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.project_root = Path(self.temporary_directory.name)
        self.annotation_root = self.project_root / ".out" / "annotation" / "compbench"
        self.annotation_root.mkdir(parents=True)

    def tearDown(self):
        self.temporary_directory.cleanup()

    def _resolve(self) -> Path:
        return resolve_annotation_file(
            project_root=self.project_root,
            benchmark="compbench",
            annotation_override_env="TEST_ANNOTATIONS",
            run_selection_env="TEST_ANNOTATION_RUN",
            filename="annotations.jsonl",
        )

    def test_defaults_to_latest_run_with_annotations(self):
        first = self.annotation_root / "run_20260101_000000_000000"
        latest = self.annotation_root / "run_20260102_000000_000000"
        incomplete = self.annotation_root / "run_20260103_000000_000000"
        for directory in (first, latest, incomplete):
            directory.mkdir()
        (first / "annotations.jsonl").write_text("first", encoding="utf-8")
        (latest / "annotations.jsonl").write_text("latest", encoding="utf-8")

        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("TEST_ANNOTATIONS", None)
            os.environ.pop("TEST_ANNOTATION_RUN", None)
            os.environ.pop("ANNOTATE_OUT", None)
            self.assertEqual(self._resolve(), (latest / "annotations.jsonl").resolve())

    def test_can_select_specific_run(self):
        selected = self.annotation_root / "run_20260101_000000_000000"
        selected.mkdir()
        path = selected / "annotations.jsonl"
        path.write_text("records", encoding="utf-8")

        with patch.dict(
            os.environ,
            {"TEST_ANNOTATION_RUN": selected.name},
            clear=False,
        ):
            os.environ.pop("TEST_ANNOTATIONS", None)
            os.environ.pop("ANNOTATE_OUT", None)
            self.assertEqual(self._resolve(), path.resolve())

    def test_explicit_annotation_file_takes_precedence(self):
        explicit = self.project_root / "custom.jsonl"
        explicit.write_text("records", encoding="utf-8")

        with patch.dict(
            os.environ,
            {"TEST_ANNOTATIONS": str(explicit)},
            clear=False,
        ):
            os.environ.pop("ANNOTATE_OUT", None)
            os.environ.pop("TEST_ANNOTATION_RUN", None)
            self.assertEqual(self._resolve(), explicit.resolve())

    def test_uses_legacy_flat_file_when_no_runs_exist(self):
        legacy = self.annotation_root / "annotations.jsonl"
        legacy.write_text("records", encoding="utf-8")

        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("TEST_ANNOTATIONS", None)
            os.environ.pop("TEST_ANNOTATION_RUN", None)
            os.environ.pop("ANNOTATE_OUT", None)
            self.assertEqual(self._resolve(), legacy.resolve())


if __name__ == "__main__":
    unittest.main()
