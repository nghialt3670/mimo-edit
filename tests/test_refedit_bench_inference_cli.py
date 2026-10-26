"""Unit tests for the RefEdit-Bench mixed-prompt inference CLI."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.refedit_bench_inference_mixed import _parse_args, _resolve_launch_path


class RefEditBenchInferenceCliTests(unittest.TestCase):
    def test_defaults_to_full_run_from_first_annotation(self):
        with patch.object(sys, "argv", ["refedit_bench_inference_mixed"]):
            args = _parse_args()

        self.assertIsNone(args.annotations)
        self.assertIsNone(args.annotation_run)
        self.assertIsNone(args.limit)
        self.assertEqual(args.start_index, 0)
        self.assertFalse(args.resume)
        self.assertIsNone(args.run)
        self.assertFalse(args.rerun_completed)
        self.assertFalse(args.rerun_spatial_selector_issues)

    def test_accepts_smoke_test_and_resume_options(self):
        with patch.object(
            sys,
            "argv",
            [
                "refedit_bench_inference_mixed",
                "--limit",
                "3",
                "--start-index",
                "2",
                "--resume",
                "--run",
                "run_20261012_012511_868075",
                "--rerun-completed",
                "--require-cuda",
            ],
        ):
            args = _parse_args()

        self.assertEqual(args.limit, 3)
        self.assertEqual(args.start_index, 2)
        self.assertTrue(args.resume)
        self.assertEqual(args.run, "run_20261012_012511_868075")
        self.assertTrue(args.rerun_completed)
        self.assertFalse(args.rerun_spatial_selector_issues)
        self.assertTrue(args.require_cuda)

    def test_accepts_rerun_spatial_selector_issues(self):
        with patch.object(
            sys,
            "argv",
            [
                "refedit_bench_inference_mixed",
                "--resume",
                "--rerun-spatial-selector-issues",
            ],
        ):
            args = _parse_args()

        self.assertTrue(args.resume)
        self.assertTrue(args.rerun_spatial_selector_issues)
        self.assertFalse(args.rerun_completed)

    def test_rejects_run_without_resume(self):
        with patch.object(
            sys,
            "argv",
            ["refedit_bench_inference_mixed", "--run", "run_20261012_012511_868075"],
        ):
            with self.assertRaises(SystemExit):
                _parse_args()

    def test_rejects_rerun_completed_without_resume(self):
        with patch.object(
            sys,
            "argv",
            ["refedit_bench_inference_mixed", "--rerun-completed"],
        ):
            with self.assertRaises(SystemExit):
                _parse_args()

    def test_rejects_rerun_spatial_selector_issues_without_resume(self):
        with patch.object(
            sys,
            "argv",
            ["refedit_bench_inference_mixed", "--rerun-spatial-selector-issues"],
        ):
            with self.assertRaises(SystemExit):
                _parse_args()

    def test_rejects_conflicting_rerun_modes(self):
        with patch.object(
            sys,
            "argv",
            [
                "refedit_bench_inference_mixed",
                "--resume",
                "--rerun-completed",
                "--rerun-spatial-selector-issues",
            ],
        ):
            with self.assertRaises(SystemExit):
                _parse_args()

    def test_rejects_non_positive_limit(self):
        with patch.object(
            sys, "argv", ["refedit_bench_inference_mixed", "--limit", "0"]
        ):
            with self.assertRaises(SystemExit):
                _parse_args()

    def test_resolves_relative_paths_from_bundle_launch_directory(self):
        with patch.dict("os.environ", {"EVAL_START_DIR": "/tmp/mixed-prompt-run"}):
            path = _resolve_launch_path(Path("annotations.jsonl"))

        self.assertEqual(path, Path("/tmp/mixed-prompt-run/annotations.jsonl").resolve())

    def test_general_entrypoint_dispatches_to_mixed_by_default(self):
        from scripts import refedit_bench_inference

        with (
            patch.object(sys, "argv", ["refedit_bench_inference", "--limit", "1"]),
            patch("scripts.refedit_bench_inference_mixed.main") as mock_mixed,
        ):
            refedit_bench_inference.main()
            mock_mixed.assert_called_once()

    def test_general_entrypoint_dispatches_to_text(self):
        from scripts import refedit_bench_inference

        with (
            patch.object(
                sys,
                "argv",
                ["refedit_bench_inference", "--prompt-mode", "text", "--limit", "1"],
            ),
            patch("scripts.refedit_bench_inference_text.main") as mock_text,
        ):
            refedit_bench_inference.main()
            mock_text.assert_called_once()


if __name__ == "__main__":
    unittest.main()
