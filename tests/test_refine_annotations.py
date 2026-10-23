from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from PIL import Image

from annotation.refine import (
    apply_refinement,
    include_identifying_tail,
    refine_annotations,
    refined_audit_path,
    refined_output_path,
    select_sam3_mask,
)
from annotation.schema import RefinementDecision


def _record() -> dict:
    return {
        "sample_id": "easy:042",
        "benchmark": "refedit_bench",
        "split": "easy",
        "original_instruction": (
            "Remove the sunglasses from the man standing on the road wearing glasses."
        ),
        "mixed_instruction": "Remove the sunglasses from <v1>.",
        "references": [
            {
                "token": "<v1>",
                "entity_id": "e1",
                "span": "the man standing on the road wearing glasses",
                "start": 30,
                "end": 73,
                "role": "anchor",
                "grounding_query": "man wearing glasses",
                "region": {
                    "bbox_xyxy": [1, 1, 9, 9],
                    "image_size": [10, 10],
                    "confidence": 1.0,
                    "label": "man wearing glasses",
                    "source": "benchmark_mask:object_instance",
                    "mask_path": "assets/refedit_bench_mixed/easy_042/e1_mask.png",
                },
            }
        ],
        "source_ref": "hf://bpathir1/RefEdit-Bench/test/42/input_image",
        "metadata": {"row_index": 42},
    }


class _Parser:
    def __init__(self, decision: RefinementDecision):
        self.decision = decision
        self.calls = 0

    def propose_refinement(self, **_kwargs) -> RefinementDecision:
        self.calls += 1
        return self.decision


class RefineAnnotationTests(unittest.TestCase):
    def test_refined_paths_use_prefix(self):
        source = Path("/tmp/run/refedit_bench_mixed.jsonl")
        self.assertEqual(
            refined_output_path(source).name, "refined_refedit_bench_mixed.jsonl"
        )
        self.assertEqual(
            refined_audit_path(source).name, "refined_refedit_bench_audit.jsonl"
        )

    def test_apply_refinement_tokens_the_edited_accessory(self):
        image = Image.new("RGB", (10, 10), "white")
        mask = Image.new("L", (10, 10), 0)
        mask.paste(255, (2, 2, 5, 5))
        rebuilt = apply_refinement(
            _record(),
            image,
            mask,
            RefinementDecision(
                optimize=True,
                reason="mask covers the man, not the sunglasses",
                span="the sunglasses",
                role="target",
                grounding_query="sunglasses",
                sam3_text="sunglasses",
            ),
        )
        self.assertEqual(rebuilt.mixed_instruction, "Remove <v1>.")
        self.assertEqual(
            rebuilt.references[0].span,
            "the sunglasses from the man standing on the road wearing glasses",
        )
        self.assertEqual(rebuilt.references[0].role, "target")
        self.assertEqual(rebuilt.references[0].region.source, "sam3:text")
        self.assertTrue(rebuilt.metadata["refined"])
        self.assertTrue(rebuilt.validation["pass"])

    def test_identifying_tail_stops_before_requested_change(self):
        text = "Change the color of the shirt of the man to red."
        start, end = text.index("the shirt"), text.index("the shirt") + len("the shirt")
        start, end = include_identifying_tail(text, start, end)
        self.assertEqual(text[start:end], "the shirt of the man")

    def test_select_sam3_mask_prefers_overlap_with_host(self):
        host = Image.new("L", (8, 8), 0)
        host.paste(255, (0, 0, 4, 8))
        inside = Image.new("L", (8, 8), 0)
        inside.paste(255, (0, 0, 2, 2))
        outside = Image.new("L", (8, 8), 0)
        outside.paste(255, (6, 0, 8, 2))
        chosen = select_sam3_mask([outside, inside], host)
        self.assertTrue(np.array_equal(np.asarray(chosen), np.asarray(inside)))

    def test_refine_copies_unoptimized_records(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            annotations = root / "refedit_bench_mixed.jsonl"
            assets = root / "assets" / "refedit_bench_mixed" / "easy_042"
            assets.mkdir(parents=True)
            mask = Image.new("L", (10, 10), 0)
            mask.paste(255, (1, 1, 9, 9))
            mask.save(assets / "e1_mask.png")
            annotations.write_text(json.dumps(_record()) + "\n", encoding="utf-8")
            image = Image.new("RGB", (10, 10), "white")
            parser = _Parser(
                RefinementDecision(optimize=False, reason="already the edit target")
            )

            counts = refine_annotations(
                annotations,
                parser=parser,
                load_image=lambda _record: image,
                segment_text=lambda _image, _text: [],
            )
            output = refined_output_path(annotations)
            copied = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(counts["copied"], 1)
            self.assertEqual(counts["optimized"], 0)
            self.assertEqual(
                copied["mixed_instruction"], _record()["mixed_instruction"]
            )
            self.assertEqual(parser.calls, 1)

    def test_refine_writes_sam3_assets_for_optimized_samples(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            annotations = root / "refedit_bench_mixed.jsonl"
            assets = root / "assets" / "refedit_bench_mixed" / "easy_042"
            assets.mkdir(parents=True)
            host = Image.new("L", (10, 10), 0)
            host.paste(255, (1, 1, 9, 9))
            host.save(assets / "e1_mask.png")
            annotations.write_text(json.dumps(_record()) + "\n", encoding="utf-8")
            image = Image.new("RGB", (10, 10), "white")
            glasses = Image.new("L", (10, 10), 0)
            glasses.paste(255, (2, 2, 5, 4))
            parser = _Parser(
                RefinementDecision(
                    optimize=True,
                    reason="mask is the man",
                    span="the sunglasses",
                    role="target",
                    grounding_query="sunglasses",
                    sam3_text="sunglasses",
                )
            )

            counts = refine_annotations(
                annotations,
                parser=parser,
                load_image=lambda _record: image,
                segment_text=lambda _image, _text: [glasses],
            )
            output = refined_output_path(annotations)
            refined = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(counts["optimized"], 1)
            self.assertEqual(refined["mixed_instruction"], "Remove <v1>.")
            mask_path = root / refined["references"][0]["region"]["mask_path"]
            crop_path = root / refined["references"][0]["region"]["crop_path"]
            self.assertTrue(mask_path.is_file())
            self.assertTrue(crop_path.is_file())
            self.assertIn("refined_refedit_bench_mixed", str(mask_path))
            self.assertIn("visual_prompts", refined["references"][0])


if __name__ == "__main__":
    unittest.main()
