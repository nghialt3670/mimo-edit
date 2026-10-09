import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from PIL import Image

from evaluation.common import (
    build_agent_context,
    build_agent_task,
    load_sample,
    run_benchmark,
    select_reference_prompts,
)
from evaluation.refedit_bench_metrics import (
    crop_pair_to_mask_roi,
    decode_refedit_bench_mask,
    summarize_viescore_scores,
)
from evaluation.refedit_bench_viescore import (
    collect_saved_outputs,
    evaluate_saved_refedit_bench_outputs,
)


class RefEditMetricTests(unittest.TestCase):
    def test_decodes_serialized_polygons_into_binary_mask(self):
        mask = decode_refedit_bench_mask(
            "[[2, 1, 5, 1, 5, 4, 2, 4]]",
            size=(8, 6),
        )

        self.assertEqual(mask.mode, "L")
        self.assertEqual(mask.getbbox(), (2, 1, 6, 5))
        self.assertEqual(mask.getpixel((3, 2)), 255)
        self.assertEqual(mask.getpixel((0, 0)), 0)

    def test_crops_both_images_to_identical_tight_gt_mask_roi(self):
        source = Image.new("RGB", (8, 6), "red")
        edited = Image.new("RGB", (4, 3), "blue")
        polygon = [[2, 1, 5, 1, 5, 4, 2, 4]]

        source_roi, edited_roi, box = crop_pair_to_mask_roi(
            source, edited, polygon
        )

        self.assertEqual(box, (2, 1, 6, 5))
        self.assertEqual(source_roi.size, (4, 4))
        self.assertEqual(edited_roi.size, source_roi.size)
        self.assertEqual(source_roi.getpixel((0, 0)), (255, 0, 0))
        self.assertEqual(edited_roi.getpixel((0, 0)), (0, 0, 255))

    def test_empty_or_invalid_mask_is_reported(self):
        with self.assertRaisesRegex(ValueError, "mask is empty"):
            crop_pair_to_mask_roi(
                Image.new("RGB", (4, 4)),
                Image.new("RGB", (4, 4)),
                [],
            )
        with self.assertRaisesRegex(ValueError, "at least three"):
            decode_refedit_bench_mask([[0, 0, 1, 1]], size=(4, 4))

    def test_viescore_summary_averages_easy_and_hard_separately(self):
        summary = summarize_viescore_scores(
            [
                {"category": "Easy", "sc": 8, "pq": 6, "o": 7},
                {"category": "easy", "sc": 6, "pq": 8, "o": 7},
                {"category": "Hard", "sc": 4, "pq": 2, "o": 3},
            ]
        )

        self.assertEqual(summary["easy"], {"sc": 7.0, "pq": 7.0, "o": 7.0, "count": 2})
        self.assertEqual(summary["hard"], {"sc": 4.0, "pq": 2.0, "o": 3.0, "count": 1})

    def test_saved_output_discovery_recovers_images_missing_from_overwritten_summary(self):
        records = [
            {"sample_id": f"easy:{index:03d}"} for index in range(3)
        ]
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            results_path = root / "results" / "results.jsonl"
            images_dir = results_path.parent / "images"
            images_dir.mkdir(parents=True)
            results_path.write_text(
                json.dumps(
                    {
                        "annotation_index": 1,
                        "sample_id": "easy:001",
                        "status": "completed",
                        "output_path": str(images_dir / "000001_easy_001_output.png"),
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            for index in range(3):
                (images_dir / f"{index:06d}_easy_{index:03d}_output.png").touch()

            outputs = collect_saved_outputs(records, results_path, images_dir)

        self.assertEqual(
            [row["annotation_index"] for row in outputs],
            [0, 1, 2],
        )
        self.assertEqual(outputs[1]["discovered_from"], "results.jsonl")

    def test_evaluator_scores_all_saved_images_without_network_or_agent_rerun(self):
        records = [
            {
                "sample_id": f"{category}:{index:03d}",
                "source_ref": f"hf://dataset/test/{index}/input_image",
                "metadata": {
                    "category": category,
                    "edit_instruction_single": f"Edit item {index}.",
                },
            }
            for index, category in enumerate(("easy", "hard", "easy"))
        ]

        class FakeImageLoader:
            def parse_reference(self, reference):
                parts = reference.removeprefix("hf://").split("/")
                return "/".join(parts[:-3]), parts[-3], int(parts[-2]), parts[-1]

            def get_row(self, dataset_id, split, row_index):
                return {
                    "category": records[row_index]["metadata"]["category"],
                    "edit_instruction_single": records[row_index]["metadata"][
                        "edit_instruction_single"
                    ],
                    "input_image": Image.new("RGB", (8, 8), "red"),
                    "mask": [[1, 1, 5, 1, 5, 5, 1, 5]],
                }

            def as_image(self, value):
                return value

        class FakeScorer:
            calls = 0

            def evaluate(self, images, instruction, *, extract_all_score):
                self.calls += 1
                self.assert_image_pair(images)
                return [8.0, 6.0, 6.9]

            @staticmethod
            def assert_image_pair(images):
                assert len(images) == 2
                assert images[0].size == images[1].size
                assert images[0].size == (5, 5)

        with tempfile.TemporaryDirectory() as temporary_directory:
            output_dir = Path(temporary_directory)
            images_dir = output_dir / "results" / "images"
            images_dir.mkdir(parents=True)
            for index, record in enumerate(records):
                output_path = (
                    images_dir
                    / f"{index:06d}_{record['sample_id'].replace(':', '_')}_output.png"
                )
                Image.new("RGB", (8, 8), "blue").save(output_path)
            # The run summary has only the middle sample; old images remain on disk.
            (output_dir / "results" / "results.jsonl").write_text(
                json.dumps(
                    {
                        "annotation_index": 1,
                        "sample_id": records[1]["sample_id"],
                        "status": "completed",
                        "output_path": str(
                            images_dir / "000001_hard_001_output.png"
                        ),
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            scorer = FakeScorer()

            rows, summary = evaluate_saved_refedit_bench_outputs(
                records,
                output_dir=output_dir,
                image_loader=FakeImageLoader(),
                scorer=scorer,
            )

            self.assertEqual(len(rows), 3)
            self.assertEqual(scorer.calls, 3)
            self.assertEqual(summary["easy"]["count"], 2)
            self.assertEqual(summary["hard"]["count"], 1)
            self.assertEqual(
                len(
                    (output_dir / "metrics" / "modified_viescore.jsonl")
                    .read_text(encoding="utf-8")
                    .splitlines()
                ),
                3,
            )
            self.assertFalse((output_dir / "results" / "metrics").exists())


class _FakeAgent:
    last_instructions = None
    last_task = None
    last_verbosity_level = None

    def __init__(self, **kwargs):
        type(self).last_instructions = kwargs.get("instructions")
        type(self).last_verbosity_level = kwargs.get("verbosity_level")
        self.trajectory = [{"step": "observed"}]
        self.trajectory_text = "observed"
        self.raw_steps = [{"raw": "step"}]
        self.variables = {
            "image_0": Image.new("RGB", (4, 4), "blue"),
            "mask_0": Image.new("L", (4, 4), 255),
            "box_0": [0, 0, 3, 3],
            "scores": np.array([0.1, 0.9], dtype=np.float32),
        }

    def send(self, task, *, objects):
        type(self).last_task = task
        message = SimpleNamespace(
            objects=[Image.new("RGB", (4, 4), "red")],
            text="Edited image",
        )
        return SimpleNamespace(message=message, raw={"result": "raw"})


class EvaluationOutputLayoutTests(unittest.TestCase):
    def test_benchmark_outputs_are_split_between_logs_and_results(self):
        record = {"sample_id": "sample:001", "instruction": "Make it red."}
        sample = {
            "instruction": "Make it red.",
            "source_image": Image.new("RGB", (4, 4), "blue"),
            "target_image": None,
            "references": [
                {
                    "token": "<v1>",
                    "role": "subject",
                    "span": "the subject",
                    "mask": None,
                    "bbox_xyxy": [0, 0, 3, 3],
                    "point_xy": [2, 2],
                    "scribble_strokes": [[[0, 0], [3, 3]]],
                    "crop": None,
                }
            ],
        }
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_dir = Path(temporary_directory)
            with (
                patch("evaluation.common.load_sample", return_value=sample),
                patch("framework.core.Agent", _FakeAgent),
                patch("framework.core.local_executor", return_value=object()),
            ):
                results = run_benchmark(
                    [record],
                    [0],
                    benchmark="refedit_bench",
                    output_dir=output_dir,
                    image_loader=object(),
                    model=SimpleNamespace(model_id="test-model"),
                    tools=[],
                    seed=2718,
                )

            sample_logs = output_dir / "logs" / "000000_sample_001"
            result_images = output_dir / "results" / "images"
            results_file = output_dir / "results" / "results.jsonl"
            self.assertTrue((sample_logs / "trajectory.txt").is_file())
            self.assertTrue((sample_logs / "trajectory.json").is_file())
            self.assertTrue((sample_logs / "variables.json").is_file())
            self.assertTrue((sample_logs / "raw.json").is_file())
            self.assertTrue((result_images / "000000_sample_001_output.png").is_file())
            self.assertTrue((result_images / "000000_sample_001_source.png").is_file())
            self.assertTrue((result_images / "000000_sample_001_annotated.png").is_file())
            self.assertTrue(results_file.is_file())
            self.assertFalse((output_dir / "metrics").exists())

            saved_result = json.loads(results_file.read_text(encoding="utf-8"))
            self.assertEqual(results[0]["status"], "completed")
            self.assertEqual(_FakeAgent.last_verbosity_level, 0)
            self.assertEqual(results[0]["random_seed"], 2718)
            self.assertEqual(_FakeAgent.last_task, "Make it red.")
            self.assertNotIn("Reference", _FakeAgent.last_task)
            self.assertIn("Reference <v1> refers to 'the subject'", _FakeAgent.last_instructions)
            self.assertIn("Never use annotation masks or crops", _FakeAgent.last_instructions)
            self.assertEqual(len(saved_result["reference_prompts"]), 1)
            self.assertIn(
                saved_result["reference_prompts"][0]["type"],
                {"box", "point", "scribble"},
            )
            self.assertEqual(saved_result["log_dir"], str(sample_logs))
            self.assertEqual(saved_result["trajectory_text_path"], str(sample_logs / "trajectory.txt"))
            self.assertEqual(saved_result["variables_path"], str(sample_logs / "variables.json"))
            self.assertEqual(saved_result["raw_path"], str(sample_logs / "raw.json"))
            self.assertEqual(saved_result["output_path"], str(result_images / "000000_sample_001_output.png"))
            saved_variables = json.loads(
                (sample_logs / "variables.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                saved_variables["image_0"]["__type__"], "PIL.Image"
            )
            self.assertEqual(
                saved_variables["mask_0"]["__type__"], "PIL.Image"
            )
            self.assertTrue(Path(saved_variables["image_0"]["path"]).is_file())
            self.assertTrue(Path(saved_variables["mask_0"]["path"]).is_file())
            self.assertEqual(saved_variables["box_0"], [0, 0, 3, 3])
            self.assertEqual(saved_variables["scores"]["shape"], [2])
            self.assertTrue(Path(saved_variables["scores"]["path"]).is_file())
            saved_raw = json.loads(
                (sample_logs / "raw.json").read_text(encoding="utf-8")
            )
            self.assertIn("agent_variables", saved_raw)
            self.assertEqual(saved_raw["random_seed"], 2718)

    def test_results_jsonl_is_flushed_after_each_sample(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_dir = Path(temporary_directory)
            results_file = output_dir / "results" / "results.jsonl"
            first_result = {
                "sample_id": "first",
                "status": "completed",
                "elapsed_seconds": 1,
            }
            second_result = {
                "sample_id": "second",
                "status": "failed",
                "elapsed_seconds": 2,
            }
            calls = 0

            class RecordingProgress:
                def __init__(self, iterable):
                    self.iterable = iterable
                    self.postfixes = []

                def __iter__(self):
                    return iter(self.iterable)

                def set_postfix(self, **values):
                    self.postfixes.append(values)

            def run_one(*args, **kwargs):
                nonlocal calls
                calls += 1
                if calls == 2:
                    lines = results_file.read_text(encoding="utf-8").splitlines()
                    self.assertEqual(len(lines), 1)
                    self.assertEqual(json.loads(lines[0])["sample_id"], "first")
                return first_result if calls == 1 else second_result

            progress = RecordingProgress([0, 1])
            with (
                patch("evaluation.common.run_one", side_effect=run_one),
                patch("evaluation.common.tqdm", return_value=progress) as tqdm_mock,
            ):
                tqdm_mock.write = lambda *_args, **_kwargs: None
                results = run_benchmark(
                    [{"sample_id": "first"}, {"sample_id": "second"}],
                    [0, 1],
                    benchmark="refedit_bench",
                    output_dir=output_dir,
                    image_loader=object(),
                    model=object(),
                    tools=[],
                )

            lines = results_file.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines), 2)
            self.assertEqual([json.loads(line)["sample_id"] for line in lines], ["first", "second"])
            self.assertEqual(results, [first_result, second_result])
            self.assertEqual(
                progress.postfixes,
                [
                    {"sample": "first", "status": "running", "refresh": True},
                    {"sample": "first", "status": "completed", "refresh": False},
                    {"sample": "second", "status": "running", "refresh": True},
                    {"sample": "second", "status": "failed", "refresh": False},
                ],
            )

    def test_agent_task_passes_exactly_one_visual_prompt_per_reference(self):
        sample = {
            "instruction": "Make it red.",
            "source_image": Image.new("RGB", (4, 4), "blue"),
            "references": [
                {
                    "token": "<v1>",
                    "role": "subject",
                    "mask": None,
                    "bbox_xyxy": [0, 0, 3, 3],
                    "point_xy": [2, 2],
                    "scribble_strokes": [[[0, 0], [3, 3]]],
                    "crop": None,
                }
            ],
        }
        select_reference_prompts(sample, seed=1)
        task, objects = build_agent_task(sample)
        context = build_agent_context(sample)

        selectors = {"box_0", "point_0", "scribble_0"} & objects.keys()
        self.assertEqual(len(selectors), 1)
        self.assertEqual(task, "Make it red.")
        selected_variable = next(iter(selectors))
        self.assertIn(f"selector variable {selected_variable}", context)

    def test_agent_context_requires_actionable_edit_object_prompts(self):
        context = build_agent_context({"references": []})

        self.assertIn("not just the desired category", context)
        self.assertIn("Transform the selected man into a woman", context)
        self.assertIn("regional for identity/shape/accessory changes", context)

    def test_visual_prompt_selection_is_reproducible_per_sample_and_seed(self):
        def make_sample():
            return {
                "sample_id": "sample-001",
                "references": [
                    {
                        "bbox_xyxy": [0, 0, 3, 3],
                        "point_xy": [2, 2],
                        "scribble_strokes": [[[0, 0], [3, 3]]],
                    },
                    {
                        "bbox_xyxy": [1, 1, 4, 4],
                        "point_xy": [3, 3],
                        "scribble_strokes": [[[1, 1], [4, 4]]],
                    },
                ],
            }

        first = make_sample()
        second = make_sample()
        select_reference_prompts(first, seed=123)
        select_reference_prompts(second, seed=123)
        self.assertEqual(
            [ref["selected_prompt"] for ref in first["references"]],
            [ref["selected_prompt"] for ref in second["references"]],
        )

    def test_compbench_add_destinations_only_sample_boxes(self):
        image = Image.new("RGB", (8, 8), "blue")

        class FakeImageLoader:
            def load_pair(self, record):
                return image, None

        for task_name in ("add", "multi_object_add"):
            with self.subTest(task=task_name):
                sample = load_sample(
                    {
                        "sample_id": "sample-001",
                        "metadata": {"task": task_name},
                        "references": [
                            {
                                "token": "<v1>",
                                "span": "the upper left",
                                "visual_prompts": {
                                    "box": {"xyxy": [1, 2, 6, 7]},
                                    "point": {"xy": [3, 4]},
                                    "scribble": {"strokes": [[[2, 3], [4, 5]]]},
                                },
                            }
                        ],
                    },
                    benchmark="compbench",
                    image_loader=FakeImageLoader(),
                )
                select_reference_prompts(sample, seed=123)
                task, objects = build_agent_task(sample)
                context = build_agent_context(sample)

                self.assertIsNone(task)
                self.assertIn("box_0", objects)
                self.assertNotIn("point_0", objects)
                self.assertNotIn("scribble_0", objects)
                self.assertEqual(
                    sample["references"][0]["selected_prompt"]["type"], "box"
                )
                self.assertIn("not an entity to select", context)
                self.assertIn("directly as the box for generate_object", context)

    def test_sample_loading_excludes_masks_and_crops_but_keeps_grounding_and_prompts(self):
        image = Image.new("RGB", (8, 8), "blue")

        class FakeImageLoader:
            def load_pair(self, record):
                return image, None

        record = {
            "sample_id": "sample",
            "original_instruction": "Make the quilt black.",
            "mixed_instruction": "Make <v1> black.",
            "source_ref": "hf://dataset/test/0/input_image",
            "target_ref": None,
            "references": [
                {
                    "token": "<v1>",
                    "role": "target",
                    "span": "the quilt",
                    "region": {
                        "bbox_xyxy": [1, 2, 7, 8],
                        "mask_path": "mask.png",
                        "crop_path": "crop.png",
                    },
                    "visual_prompts": {
                        "box": {"xyxy": [1, 2, 7, 8]},
                        "point": {"xy": [3, 4]},
                        "scribble": {"strokes": [[[2, 3], [4, 5]]]},
                    },
                }
            ],
        }
        sample = load_sample(
            record,
            benchmark="refedit_bench",
            image_loader=FakeImageLoader(),
        )

        ref = sample["references"][0]
        self.assertEqual(sample["instruction"], "Make the quilt black.")
        self.assertEqual(ref["span"], "the quilt")
        self.assertEqual(ref["bbox_xyxy"], [1, 2, 7, 8])
        self.assertEqual(ref["point_xy"], [3, 4])
        self.assertEqual(ref["scribble_strokes"], [[[2, 3], [4, 5]]])
        self.assertNotIn("mask", ref)
        self.assertNotIn("crop", ref)
