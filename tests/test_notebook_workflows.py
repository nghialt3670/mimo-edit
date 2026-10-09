import os
import tempfile
import unittest
from pathlib import Path
from types import ModuleType
from unittest.mock import patch

from annotation.workflows import (
    AnnotationConfig,
    comma_separated_env,
    run_hf_annotation,
    selected_rows,
)
from evaluation.notebook import (
    create_agent_model,
    create_run_directory,
    load_benchmark,
    resolve_run_directory,
)
from framework.utils.project import find_project_root


class AnnotationWorkflowTests(unittest.TestCase):
    def test_config_reads_shared_annotation_settings(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            with (
                patch("annotation.workflows.load_project_environment"),
                patch.dict(
                    os.environ,
                    {
                        "ANNOTATE_LIMIT": "0",
                        "ANNOTATE_STREAMING": "0",
                        "ANNOTATE_BOX_THRESHOLD": "0.4",
                        "ANNOTATE_OUT": temporary_directory,
                        "HF_TOKEN": "test-token",
                    },
                    clear=True,
                ),
            ):
                config = AnnotationConfig.from_env("compbench")

        self.assertIsNone(config.limit)
        self.assertFalse(config.streaming)
        self.assertEqual(config.box_threshold, 0.4)
        self.assertEqual(config.output_dir, Path(temporary_directory).resolve())
        self.assertEqual(config.hf_token, "test-token")

    def test_config_resolves_default_output_path_from_automatic_project_root(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            project_root = Path(temporary_directory)
            with (
                patch(
                    "annotation.workflows.load_project_environment",
                    return_value=project_root,
                ),
                patch.dict(os.environ, {}, clear=True),
            ):
                config = AnnotationConfig.from_env("refedit_bench")

        self.assertEqual(
            config.output_dir,
            (project_root / ".out/annotation/refedit_bench").resolve(),
        )

    def test_config_rejects_negative_limit_and_out_of_range_threshold(self):
        with patch("annotation.workflows.load_project_environment"):
            with patch.dict(os.environ, {"ANNOTATE_LIMIT": "-1"}, clear=True):
                with self.assertRaisesRegex(ValueError, "ANNOTATE_LIMIT"):
                    AnnotationConfig.from_env("compbench")
            with patch.dict(
                os.environ,
                {"ANNOTATE_BOX_THRESHOLD": "1.1"},
                clear=True,
            ):
                with self.assertRaisesRegex(ValueError, "thresholds"):
                    AnnotationConfig.from_env("compbench")

    def test_comma_separated_env_and_filtered_rows_keep_original_indices(self):
        with patch.dict(os.environ, {"TASKS": " add,remove,,add "}, clear=True):
            tasks = comma_separated_env("TASKS", "")

        rows = [
            {"task": "skip"},
            {"task": "add"},
            {"task": "remove"},
        ]
        kept = list(selected_rows(rows, lambda row: row["task"] in tasks))

        self.assertEqual(tasks, {"add", "remove"})
        self.assertEqual([row["_row_index"] for row in kept], [1, 2])

    def test_hf_workflow_runs_converter_after_validating_dataset_schema(self):
        class FakeDataset:
            column_names = ["task", "image"]

            def __iter__(self):
                return iter([{"task": "add", "image": "asset"}])

        dataset = FakeDataset()
        dataset_module = ModuleType("datasets")
        dataset_module.load_dataset = lambda *args, **kwargs: dataset
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_dir = Path(temporary_directory) / "annotations"
            config = AnnotationConfig(output_dir=output_dir)
            with (
                patch.dict("sys.modules", {"datasets": dataset_module}),
                patch.object(AnnotationConfig, "create_annotator", return_value=object()),
                patch(
                    "annotation.workflows.convert_rows",
                    return_value={"accepted": 1},
                ) as convert,
            ):
                summary = run_hf_annotation(
                    dataset_id="test/dataset",
                    split="train",
                    adapter=lambda row, split, index: row,
                    config=config,
                    output_name="output.jsonl",
                    audit_name="audit.jsonl",
                    expected_columns={"task", "image"},
                    use_grounder=False,
                )

        self.assertEqual(summary, {"accepted": 1})
        convert.assert_called_once()
        self.assertEqual(
            convert.call_args.kwargs["output"],
            output_dir / "output.jsonl",
        )
        with patch.dict("sys.modules", {"datasets": dataset_module}):
            with self.assertRaisesRegex(ValueError, "missing columns"):
                run_hf_annotation(
                    dataset_id="test/dataset",
                    split="train",
                    adapter=lambda row, split, index: row,
                    config=config,
                    output_name="output.jsonl",
                    audit_name="audit.jsonl",
                    expected_columns={"missing"},
                    use_grounder=False,
                )
        self.assertEqual(convert.call_count, 1)


class EvaluationNotebookWorkflowTests(unittest.TestCase):
    def test_project_root_falls_back_to_installed_source_location(self):
        self.assertEqual(
            find_project_root(Path("/")),
            Path(__file__).resolve().parents[1],
        )

    def test_create_agent_model_uses_common_openai_compatible_settings(self):
        class FakeOpenAIModel:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

        smolagents = ModuleType("smolagents")
        smolagents.OpenAIModel = FakeOpenAIModel
        with (
            patch("evaluation.notebook.load_project_environment"),
            patch.dict(
                os.environ,
                {
                    "MMEDIT_API_KEY": "test-api-key",
                    "MMEDIT_MODEL_ID": "test-model",
                    "MMEDIT_OPENAI_BASE_URL": "https://example.test/v1",
                },
                clear=True,
            ),
            patch.dict("sys.modules", {"smolagents": smolagents}),
        ):
            model = create_agent_model()

        self.assertEqual(
            model.kwargs,
            {
                "model_id": "test-model",
                "api_base": "https://example.test/v1",
                "api_key": "test-api-key",
                "temperature": 0,
            },
        )

    def test_load_benchmark_resolves_paths_and_reads_annotations(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            project_root = Path(temporary_directory)
            annotation_file = project_root / "annotations.jsonl"
            annotation_file.write_text('{"sample_id":"sample-1"}\n', encoding="utf-8")
            output_dir = project_root / "outputs"
            with (
                patch("evaluation.notebook.load_project_environment", return_value=project_root),
                patch.dict(
                    os.environ,
                    {
                        "MMEDIT_COMPBENCH_ANNOTATIONS": str(annotation_file),
                        "MMEDIT_COMPBENCH_OUTPUT_DIR": str(output_dir),
                    },
                    clear=True,
                ),
            ):
                context = load_benchmark("compbench")

        self.assertEqual(context.project_root, project_root)
        self.assertEqual(context.annotation_file, annotation_file.resolve())
        self.assertEqual(context.output_dir, output_dir.resolve())
        self.assertEqual(context.records, [{"sample_id": "sample-1"}])
        self.assertFalse(output_dir.exists())

    def test_load_benchmark_rejects_unknown_benchmark(self):
        with patch("evaluation.notebook.load_project_environment"):
            with self.assertRaisesRegex(ValueError, "Unsupported benchmark"):
                load_benchmark("unknown")

    def test_load_benchmark_uses_out_directory_by_default(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            project_root = Path(temporary_directory)
            annotation_file = (
                project_root
                / ".out/annotation/compbench/compbench_local_multiobject_mixed.jsonl"
            )
            annotation_file.parent.mkdir(parents=True)
            annotation_file.write_text('{"sample_id":"sample-1"}\n', encoding="utf-8")
            with (
                patch(
                    "evaluation.notebook.load_project_environment",
                    return_value=project_root,
                ),
                patch.dict(os.environ, {}, clear=True),
            ):
                context = load_benchmark("compbench")

        self.assertEqual(context.annotation_file, annotation_file.resolve())
        self.assertEqual(
            context.output_dir,
            (project_root / ".out/evaluation/compbench").resolve(),
        )

    def test_load_refedit_bench_uses_generated_annotation_output_by_default(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            project_root = Path(temporary_directory)
            annotation_file = (
                project_root
                / ".out/annotation/refedit_bench/refedit_bench_mixed.jsonl"
            )
            annotation_file.parent.mkdir(parents=True)
            annotation_file.write_text('{"sample_id":"sample-1"}\n', encoding="utf-8")
            with (
                patch(
                    "evaluation.notebook.load_project_environment",
                    return_value=project_root,
                ),
                patch.dict(os.environ, {}, clear=True),
            ):
                context = load_benchmark("refedit_bench")

        self.assertEqual(context.annotation_file, annotation_file.resolve())
        self.assertEqual(context.records, [{"sample_id": "sample-1"}])

    def test_create_run_directory_creates_unique_run_folders(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_root = Path(temporary_directory)
            first_run = create_run_directory(output_root)
            second_run = create_run_directory(output_root)
            self.assertRegex(first_run.name, r"^run_\d{8}_\d{6}_\d{6}(?:_\d+)?$")
            self.assertNotEqual(first_run, second_run)
            self.assertTrue(first_run.is_dir())
            self.assertTrue(second_run.is_dir())

    def test_resolve_run_directory_selects_latest_or_explicit_run(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_root = Path(temporary_directory)
            older = output_root / "run_20261001_120000_000000"
            latest = output_root / "run_20261002_120000_000000"
            for run_dir in (older, latest):
                results_dir = run_dir / "results"
                results_dir.mkdir(parents=True)
                (results_dir / "results.jsonl").touch()

            self.assertEqual(resolve_run_directory(output_root), latest.resolve())
            self.assertEqual(
                resolve_run_directory(output_root, older.name),
                older.resolve(),
            )
            with self.assertRaisesRegex(ValueError, "run_<timestamp>"):
                resolve_run_directory(output_root, "../results")


if __name__ == "__main__":
    unittest.main()
