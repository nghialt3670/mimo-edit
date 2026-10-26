from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.compbench_inference_mixed import _parse_args, _resolve_launch_path


class CompBenchInferenceCliTests(unittest.TestCase):
    def test_defaults_to_full_run_from_first_annotation(self):
        with patch.object(sys, "argv", ["compbench_inference_mixed"]):
            args = _parse_args()

        self.assertIsNone(args.annotations)
        self.assertIsNone(args.annotation_run)
        self.assertIsNone(args.limit)
        self.assertIsNone(args.tasks)
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
                "compbench_inference_mixed",
                "--limit",
                "3",
                "--tasks",
                "add",
                "multi_object_remove",
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
        self.assertEqual(args.tasks, ["add", "multi_object_remove"])
        self.assertIsNone(args.samples_file)
        self.assertEqual(args.start_index, 2)
        self.assertTrue(args.resume)
        self.assertEqual(args.run, "run_20261012_012511_868075")
        self.assertTrue(args.rerun_completed)
        self.assertTrue(args.require_cuda)

    def test_rejects_run_without_resume(self):
        with patch.object(
            sys,
            "argv",
            ["compbench_inference_mixed", "--run", "run_20261012_012511_868075"],
        ):
            with self.assertRaises(SystemExit):
                _parse_args()

    def test_rejects_rerun_completed_without_resume(self):
        with patch.object(
            sys,
            "argv",
            ["compbench_inference_mixed", "--rerun-completed"],
        ):
            with self.assertRaises(SystemExit):
                _parse_args()

    def test_rejects_non_positive_limit(self):
        with patch.object(sys, "argv", ["compbench_inference_mixed", "--limit", "0"]):
            with self.assertRaises(SystemExit):
                _parse_args()

    def test_accepts_samples_file(self):
        with patch.object(
            sys,
            "argv",
            ["compbench_inference_mixed", "--samples-file", "retry.txt"],
        ):
            args = _parse_args()

        self.assertEqual(args.samples_file, Path("retry.txt"))

    def test_rejects_samples_file_with_tasks(self):
        with patch.object(
            sys,
            "argv",
            [
                "compbench_inference_mixed",
                "--samples-file",
                "retry.txt",
                "--tasks",
                "add",
            ],
        ):
            with self.assertRaises(SystemExit):
                _parse_args()

    def test_rejects_samples_file_with_limit(self):
        with patch.object(
            sys,
            "argv",
            [
                "compbench_inference_mixed",
                "--samples-file",
                "retry.txt",
                "--limit",
                "3",
            ],
        ):
            with self.assertRaises(SystemExit):
                _parse_args()

    def test_resolves_relative_paths_from_bundle_launch_directory(self):
        with patch.dict("os.environ", {"EVAL_START_DIR": "/tmp/mixed-prompt-run"}):
            path = _resolve_launch_path(Path("annotations.jsonl"))

        self.assertEqual(path, Path("/tmp/mixed-prompt-run/annotations.jsonl").resolve())


if __name__ == "__main__":
    unittest.main()
