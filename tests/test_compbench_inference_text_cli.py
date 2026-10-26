from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.compbench_inference_text import _parse_args, _resolve_launch_path


class CompBenchTextInferenceCliTests(unittest.TestCase):
    def test_defaults_to_full_run_from_first_annotation(self):
        with patch.object(sys, "argv", ["compbench_inference_text"]):
            args = _parse_args()

        self.assertIsNone(args.annotations)
        self.assertIsNone(args.annotation_run)
        self.assertIsNone(args.limit)
        self.assertIsNone(args.samples_file)
        self.assertEqual(args.start_index, 0)
        self.assertFalse(args.resume)
        self.assertIsNone(args.run)
        self.assertFalse(args.rerun_completed)

    def test_accepts_smoke_test_and_resume_options(self):
        with patch.object(
            sys,
            "argv",
            [
                "compbench_inference_text",
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
        self.assertIsNone(args.samples_file)
        self.assertEqual(args.start_index, 2)
        self.assertTrue(args.resume)
        self.assertEqual(args.run, "run_20261012_012511_868075")
        self.assertTrue(args.rerun_completed)
        self.assertTrue(args.require_cuda)

    def test_accepts_samples_file(self):
        with patch.object(
            sys,
            "argv",
            ["compbench_inference_text", "--samples-file", "retry.txt"],
        ):
            args = _parse_args()

        self.assertEqual(args.samples_file, Path("retry.txt"))

    def test_rejects_samples_file_with_limit(self):
        with patch.object(
            sys,
            "argv",
            [
                "compbench_inference_text",
                "--samples-file",
                "retry.txt",
                "--limit",
                "3",
            ],
        ):
            with self.assertRaises(SystemExit):
                _parse_args()

    def test_rejects_run_without_resume(self):
        with patch.object(
            sys,
            "argv",
            ["compbench_inference_text", "--run", "run_20261012_012511_868075"],
        ):
            with self.assertRaises(SystemExit):
                _parse_args()

    def test_resolves_relative_paths_from_bundle_launch_directory(self):
        with patch.dict("os.environ", {"EVAL_START_DIR": "/tmp/mixed-prompt-run"}):
            path = _resolve_launch_path(Path("annotations.jsonl"))

        self.assertEqual(path, Path("/tmp/mixed-prompt-run/annotations.jsonl").resolve())


if __name__ == "__main__":
    unittest.main()
