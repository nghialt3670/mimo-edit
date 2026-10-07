import unittest
from unittest.mock import Mock, patch

from PIL import Image, ImageDraw

from inference import ModelCache, ModelConfig
from inference.editing import (
    _load_flux2_klein_image_edit_backend,
    edit_masked_object,
    semantic_edit_crop,
)
from inference.inpainting import (
    inpaint_with_configured_backend,
    inpaint_with_gligen,
    inpaint_with_stable_diffusion,
)
from tools.image_editing.apply_filter import apply_filter
from tools import image_editing_tools
from tools.image_editing.generate_object import generate_object
from tools.image_editing.segment_object import segment_object
from utils.image import masked_layer, paste_layer


class ModelConfigTests(unittest.TestCase):
    def test_masked_layer_feathers_edges_without_background_color_fringe(self):
        image = Image.new("RGB", (20, 20), "blue")
        ImageDraw.Draw(image).rectangle((5, 5, 14, 14), fill="red")
        mask = Image.new("L", image.size, 0)
        ImageDraw.Draw(mask).rectangle((5, 5, 14, 14), fill=255)

        layer, origin = masked_layer(image, mask)
        alpha = layer.getchannel("A")
        inner_edge = alpha.getpixel((3, 5))
        outer_edge = alpha.getpixel((2, 5))
        clean_background = Image.new("RGB", image.size, "green")
        composite = paste_layer(clean_background, layer, origin)

        self.assertLess(inner_edge, 255)
        self.assertGreater(inner_edge, 0)
        self.assertGreater(outer_edge, 0)
        self.assertLess(outer_edge, inner_edge)
        self.assertEqual(layer.getpixel((2, 5))[:3], (255, 0, 0))
        self.assertEqual(composite.getpixel((0, 0)), (0, 128, 0))
        self.assertNotEqual(composite.getpixel((5, 10)), (255, 0, 0))
        self.assertEqual(composite.getpixel((1, 10)), (0, 128, 0))

    def test_tool_model_override_and_defaults(self):
        config = ModelConfig(model_ids={"generate_object": "org/custom-inpaint"})

        self.assertEqual(
            config.model_id("generate_object", "default"), "org/custom-inpaint"
        )
        self.assertEqual(config.model_id("remove_object", "default"), "default")

    def test_cache_separates_different_model_ids(self):
        cache = ModelCache()
        first = object()
        second = object()

        self.assertIs(cache.get("pipeline", lambda: first, model_id="org/first"), first)
        self.assertIs(cache.get("pipeline", lambda: second, model_id="org/second"), second)
        self.assertIs(cache.get("pipeline", lambda: second, model_id="org/first"), first)

    def test_rejects_unknown_model_keys(self):
        with self.assertRaisesRegex(ValueError, "unsupported model config keys"):
            ModelConfig(model_ids={"unknown_tool": "org/model"})

    def test_model_config_validates_backend_choices(self):
        config = ModelConfig(
            backends={
                "edit_object": "flux2-klein-9b-kv",
                "generate_object_mask": "flux-fill",
                "generate_object_box": "flux-fill",
            }
        )

        self.assertEqual(config.backend("edit_object", "default"), "flux2-klein-9b-kv")
        with self.assertRaisesRegex(ValueError, "unsupported model backend keys"):
            ModelConfig(backends={"unknown_tool": "flux-fill"})
        with self.assertRaisesRegex(ValueError, "unsupported model backend choices"):
            ModelConfig(backends={"edit_object": "flux-fill"})

    def test_route_model_id_override_precedes_legacy_generate_object_override(self):
        config = ModelConfig(
            model_ids={
                "generate_object": "org/legacy",
                "generate_object_box": "org/box",
            }
        )

        self.assertEqual(config.model_id("generate_object_mask", "default"), "org/legacy")
        self.assertEqual(config.model_id("generate_object_box", "default"), "org/box")

    def test_regional_edit_composites_changes_inside_padding(self):
        source = Image.new("RGB", (64, 64), "blue")
        mask = Image.new("L", source.size, 0)
        ImageDraw.Draw(mask).rectangle((20, 20, 43, 43), fill=255)
        generated = Image.new("RGB", source.size, "red")

        with patch(
            "inference.editing.semantic_edit_crop", return_value=generated
        ):
            result = edit_masked_object(
                ModelCache(), source, mask, "add a hat", mode="regional", padding=0.25
            )

        self.assertNotEqual(result.getpixel((18, 30)), (0, 0, 255))
        self.assertNotEqual(result.getpixel((10, 30)), (0, 0, 255))
        self.assertEqual(result.getpixel((7, 30)), (0, 0, 255))
        self.assertEqual(result.getpixel((30, 30)), (255, 0, 0))

    def test_local_edit_feathers_original_mask_boundary(self):
        source = Image.new("RGB", (40, 40), "blue")
        mask = Image.new("L", source.size, 0)
        ImageDraw.Draw(mask).rectangle((10, 10, 29, 29), fill=255)
        generated = Image.new("RGB", source.size, "red")

        with patch("inference.editing.semantic_edit_crop", return_value=generated):
            result = edit_masked_object(
                ModelCache(),
                source,
                mask,
                "change its color",
                mode="local",
            )

        self.assertEqual(result.getpixel((0, 0)), (0, 0, 255))
        transition_pixel = result.getpixel((9, 20))
        self.assertGreater(transition_pixel[0], 0)
        self.assertLess(transition_pixel[0], 255)
        self.assertGreater(transition_pixel[2], 0)
        self.assertLess(transition_pixel[2], 255)
        self.assertEqual(result.getpixel((3, 20)), (0, 0, 255))
        self.assertEqual(result.getpixel((10, 20)), (255, 0, 0))
        self.assertEqual(result.getpixel((15, 20)), (255, 0, 0))

    def test_regional_edit_uses_padded_crop_and_dilated_composite_mask(self):
        source = Image.new("RGB", (120, 80), "blue")
        mask = Image.new("L", source.size, 0)
        ImageDraw.Draw(mask).rectangle((30, 20, 89, 39), fill=255)
        generated = Image.new("RGB", (90, 30), "red")

        with patch(
            "inference.editing.semantic_edit_crop", return_value=generated
        ) as edit:
            result = edit_masked_object(
                ModelCache(),
                source,
                mask,
                "add a hat",
                mode="regional",
                padding=0.25,
            )

        crop, prompt = edit.call_args.args[1:3]
        self.assertGreater(crop.width, 90)
        self.assertGreater(crop.height, 30)
        self.assertIn("add a hat", prompt)
        self.assertEqual(result.getpixel((30, 30)), (255, 0, 0))
        self.assertNotEqual(result.getpixel((15, 30)), (0, 0, 255))
        self.assertNotEqual(result.getpixel((14, 30)), (0, 0, 255))
        self.assertNotEqual(result.getpixel((60, 15)), (0, 0, 255))
        self.assertNotEqual(result.getpixel((60, 14)), (0, 0, 255))
        self.assertEqual(result.getpixel((8, 30)), (0, 0, 255))

    def test_global_edit_passes_full_image_and_returns_full_result(self):
        source = Image.new("RGB", (40, 30), "blue")
        mask = Image.new("L", source.size, 0)
        ImageDraw.Draw(mask).rectangle((10, 8, 20, 18), fill=255)
        generated = Image.new("RGB", source.size, "red")
        with patch(
            "inference.editing.semantic_edit_crop", return_value=generated
        ) as edit:
            result = edit_masked_object(
                ModelCache(), source, mask, "change the whole scene", mode="global"
            )

        self.assertEqual(edit.call_args.args[1].size, source.size)
        self.assertIn("pixel box", edit.call_args.args[2])
        self.assertEqual(result.getpixel((0, 0)), (255, 0, 0))

    def test_apply_filter_preserves_global_filtering_without_mask(self):
        image = Image.new("RGB", (10, 10), (30, 80, 140))

        result = apply_filter(None, image, "invert", 0)

        self.assertEqual(result.getpixel((0, 0)), (225, 175, 115))

    def test_apply_filter_masks_object_and_feathers_boundary_inside_mask(self):
        image = Image.new("RGB", (20, 20), (20, 40, 60))
        mask = Image.new("L", image.size, 0)
        ImageDraw.Draw(mask).rectangle((5, 5, 14, 14), fill=255)

        result = apply_filter(None, image, "invert", 0, mask)

        self.assertEqual(result.getpixel((0, 0)), image.getpixel((0, 0)))
        self.assertEqual(result.getpixel((10, 10)), (235, 215, 195))
        self.assertNotEqual(result.getpixel((5, 10)), (235, 215, 195))
        self.assertNotEqual(result.getpixel((5, 10)), image.getpixel((5, 10)))
        self.assertEqual(result.getpixel((4, 10)), image.getpixel((4, 10)))

    def test_apply_filter_rejects_empty_object_mask(self):
        with self.assertRaisesRegex(ValueError, "mask is empty"):
            apply_filter(
                None,
                Image.new("RGB", (10, 10), "blue"),
                "invert",
                0,
                Image.new("L", (10, 10), 0),
            )

    def test_edit_rejects_padding_outside_zero_to_one(self):
        source = Image.new("RGB", (20, 20), "blue")
        mask = Image.new("L", source.size, 255)

        for padding in (-0.01, 1.01):
            with self.subTest(padding=padding), self.assertRaisesRegex(
                ValueError, "padding must be between 0 and 1"
            ):
                edit_masked_object(
                    ModelCache(),
                    source,
                    mask,
                    "edit",
                    mode="regional",
                    padding=padding,
                )

    def test_public_tool_registry_uses_single_object_names_and_inputs(self):
        tools = {tool.name: tool for tool in image_editing_tools()}

        self.assertTrue(
            {
                "edit_object",
                "remove_object",
                "replace_object",
                "move_object",
                "flip_object",
                "rotate_object",
                "scale_object",
                "shift_object",
                "generate_object",
                "segment_object",
                "apply_filter",
            }.issubset(tools)
        )
        self.assertFalse(
            {
                "edit_objects",
                "remove_objects",
                "replace_objects",
                "move_objects",
                "flip_objects",
                "rotate_objects",
                "scale_objects",
                "shift_objects",
                "generate_objects",
                "paste_objects",
                "segment_objects",
                "segment_object_by_text",
                "inpaint_objects",
                "inpaint_region",
            }.intersection(tools)
        )
        self.assertIn("mask", tools["edit_object"].inputs)
        self.assertNotIn("masks", tools["edit_object"].inputs)
        self.assertIn("mode", tools["edit_object"].inputs)
        self.assertIn(
            "not just a label or category",
            tools["edit_object"].description,
        )
        self.assertIn(
            "Transform the selected man into a woman",
            tools["edit_object"].description,
        )
        self.assertIn(
            "Do not pass only a subject/category label",
            tools["edit_object"].inputs["prompt"]["description"],
        )
        self.assertIn("mask", tools["apply_filter"].inputs)
        self.assertTrue(tools["apply_filter"].inputs["mask"]["nullable"])
        for name in (
            "remove_object",
            "replace_object",
            "move_object",
            "flip_object",
            "rotate_object",
            "scale_object",
            "shift_object",
        ):
            with self.subTest(tool=name):
                self.assertIn("mask", tools[name].inputs)
                self.assertNotIn("masks", tools[name].inputs)
        self.assertNotIn("get_box", tools)
        self.assertNotIn("paste_object", tools)
        self.assertIn("mask", tools["generate_object"].inputs)
        self.assertIn("box", tools["generate_object"].inputs)
        self.assertIn("phrase", tools["generate_object"].inputs)
        for selector in ("prompt", "point", "box", "scribble"):
            self.assertIn(selector, tools["segment_object"].inputs)

    def test_generate_object_uses_shared_inpainting_route_for_mask_and_box(self):
        image = Image.new("RGB", (20, 20), "blue")
        mask = Image.new("L", image.size, 0)
        ImageDraw.Draw(mask).rectangle((5, 5, 14, 14), fill=255)
        generated = Image.new("RGB", image.size, "red")
        models = ModelCache()

        with patch(
            "tools.image_editing.generate_object.inpaint_with_configured_backend",
            return_value=generated,
        ) as inpaint:
            mask_result = generate_object(models, image, "fill this area", mask=mask)
            box_result = generate_object(
                models, image, "fill this area", box=[5, 5, 15, 15]
            )

        self.assertIs(mask_result, generated)
        self.assertIs(box_result, generated)
        self.assertEqual(inpaint.call_count, 2)
        self.assertEqual(
            [call.kwargs["route_key"] for call in inpaint.call_args_list],
            ["generate_object_mask", "generate_object_box"],
        )
        self.assertEqual(inpaint.call_args_list[1].args[2].getbbox(), (5, 5, 16, 16))

    def test_configured_generate_object_backend_selects_flux_fill(self):
        image = Image.new("RGB", (31, 29), "blue")
        mask = Image.new("L", image.size, 255)
        generated = Image.new("RGB", (32, 32), "red")
        pipeline = Mock(return_value=Mock(images=[generated]))
        models = ModelCache(
            ModelConfig(
                backends={
                    "generate_object_mask": "flux-fill",
                    "generate_object_box": "flux-fill",
                },
                model_ids={
                    "generate_object_mask": "org/custom-mask-fill",
                    "generate_object_box": "org/custom-box-fill",
                },
            )
        )

        with (
            patch("inference.inpainting.device", return_value="cpu"),
            patch("inference.inpainting._load_flux_fill", return_value=pipeline) as load,
        ):
            box_result = inpaint_with_configured_backend(
                models, image, mask, "fill this area", route_key="generate_object_box"
            )
            mask_result = inpaint_with_configured_backend(
                models, image, mask, "fill this area", route_key="generate_object_mask"
            )

        self.assertEqual(
            [call.args[0] for call in load.call_args_list],
            ["org/custom-box-fill", "org/custom-mask-fill"],
        )
        self.assertEqual(box_result.size, image.size)
        self.assertEqual(mask_result.size, image.size)
        self.assertEqual(pipeline.call_count, 2)
        kwargs = pipeline.call_args_list[0].kwargs
        self.assertEqual(kwargs["image"].size, (32, 32))
        self.assertEqual(kwargs["mask_image"].size, (32, 32))
        self.assertEqual(kwargs["prompt"], "fill this area")
        self.assertEqual(kwargs["guidance_scale"], 30)
        self.assertEqual(kwargs["num_inference_steps"], 50)

    def test_configured_generate_object_backend_preserves_stable_diffusion_default(self):
        image = Image.new("RGB", (20, 20), "blue")
        mask = Image.new("L", image.size, 255)
        generated = Image.new("RGB", image.size, "red")
        models = ModelCache()

        with patch(
            "inference.inpainting.inpaint_with_stable_diffusion",
            return_value=generated,
        ) as stable_diffusion:
            result = inpaint_with_configured_backend(
                models,
                image,
                mask,
                "fill this area",
                route_key="generate_object_mask",
            )

        self.assertIs(result, generated)
        stable_diffusion.assert_called_once()
        self.assertIs(stable_diffusion.call_args.args[0], models)
        self.assertEqual(stable_diffusion.call_args.kwargs["model_key"], "generate_object_mask")

    def test_inpainting_routes_restore_source_image_dimensions(self):
        image = Image.new("RGB", (80, 60), "blue")
        mask = Image.new("L", image.size, 255)
        generated = Image.new("RGB", (512, 512), "red")
        stable_pipeline = Mock(return_value=Mock(images=[generated]))
        gligen_pipeline = Mock(return_value=Mock(images=[generated]))
        models = ModelCache()

        with (
            patch("inference.inpainting.device", return_value="cpu"),
            patch(
                "inference.inpainting._load_stable_diffusion",
                return_value=stable_pipeline,
            ),
            patch(
                "inference.inpainting._load_gligen",
                return_value=gligen_pipeline,
            ),
        ):
            filled = inpaint_with_stable_diffusion(
                models, image, mask, "fill the region"
            )
            placed = inpaint_with_gligen(
                models, image, "a photo", ["a ball"], [[10, 10, 30, 30]]
            )

        self.assertEqual(filled.size, image.size)
        self.assertEqual(placed.size, image.size)
        self.assertEqual(
            gligen_pipeline.call_args.kwargs["gligen_scheduled_sampling_beta"],
            0.3,
        )
        self.assertEqual(
            gligen_pipeline.call_args.kwargs["gligen_boxes"],
            [[0.125, 1 / 6, 0.375, 0.5]],
        )

    def test_segment_object_text_returns_highest_scoring_mask(self):
        image = Image.new("RGB", (20, 20), "blue")
        best = Image.new("L", image.size, 255)
        other = Image.new("L", image.size, 0)
        models = ModelCache()

        with patch(
            "tools.image_editing.segment_object.segment_text",
            return_value=[best, other],
        ) as segment:
            result = segment_object(models, image, prompt="  cat  ")

        self.assertIs(result, best)
        segment.assert_called_once_with(
            models, image, "  cat  ", model_key="segment_object"
        )

    def test_segment_object_text_raises_when_no_object_matches(self):
        image = Image.new("RGB", (20, 20), "blue")

        with (
            patch("tools.image_editing.segment_object.segment_text", return_value=[]),
            self.assertRaisesRegex(ValueError, "No object matched"),
        ):
            segment_object(ModelCache(), image, prompt="cat")

    def test_segment_object_requires_exactly_one_method(self):
        image = Image.new("RGB", (20, 20), "blue")

        with self.assertRaisesRegex(ValueError, "exactly one"):
            segment_object(ModelCache(), image)
        with self.assertRaisesRegex(ValueError, "exactly one"):
            segment_object(
                ModelCache(), image, prompt="cat", point=[5, 5]
            )
        with self.assertRaisesRegex(ValueError, "non-empty string"):
            segment_object(ModelCache(), image, prompt=" ")

    def test_generate_object_rejects_ambiguous_or_missing_region(self):
        image = Image.new("RGB", (20, 20), "blue")
        mask = Image.new("L", image.size, 255)
        models = ModelCache()

        with self.assertRaisesRegex(ValueError, "Provide one mask or one box"):
            generate_object(models, image, "generate")
        with self.assertRaisesRegex(ValueError, "Provide a mask or a box"):
            generate_object(models, image, "generate", mask=mask, box=[0, 0, 10, 10])
        with self.assertRaisesRegex(ValueError, "phrase requires a box"):
            generate_object(models, image, "generate", phrase="a cat")
        with self.assertRaisesRegex(ValueError, "uses a box and phrase"):
            generate_object(
                models,
                image,
                "generate",
                mask=mask,
                box=[0, 0, 10, 10],
                phrase="a cat",
            )

    def test_default_edit_object_route_uses_flux2_klein(self):
        models = ModelCache()
        image = Image.new("RGB", (12, 12), "blue")
        mask = Image.new("L", image.size, 255)
        backend = Mock()
        backend.edit.return_value = image

        with patch(
            "inference.editing._load_flux2_klein_image_edit_backend",
            return_value=backend,
        ) as loader:
            result = semantic_edit_crop(models, image, "edit")

        loader.assert_called_once_with(
            "black-forest-labs/FLUX.2-klein-4B", "flux2-klein-4b"
        )
        backend.edit.assert_called_once_with(image, "edit")
        self.assertIs(result, image)

    def test_flux2_klein_9b_kv_selects_kv_pipeline_and_checkpoint(self):
        models = ModelCache(ModelConfig(backends={"edit_object": "flux2-klein-9b-kv"}))
        image = Image.new("RGB", (12, 12), "blue")
        backend = Mock()
        backend.edit.return_value = image

        with patch(
            "inference.editing._load_flux2_klein_image_edit_backend",
            return_value=backend,
        ) as loader:
            result = semantic_edit_crop(models, image, "edit")

        loader.assert_called_once_with(
            "black-forest-labs/FLUX.2-klein-9b-kv", "flux2-klein-9b-kv"
        )
        backend.edit.assert_called_once_with(image, "edit")
        self.assertIs(result, image)

    def test_flux2_klein_loader_uses_supported_dtype_and_model_id(self):
        import torch

        pipeline = Mock()
        pipeline.to.return_value = pipeline
        with (
            patch("inference.editing.device", return_value="mps"),
            patch(
                "diffusers.Flux2KleinPipeline.from_pretrained",
                return_value=pipeline,
            ) as load_pipeline,
        ):
            backend = _load_flux2_klein_image_edit_backend(
                "black-forest-labs/FLUX.2-klein-4B"
            )

        load_pipeline.assert_called_once_with(
            "black-forest-labs/FLUX.2-klein-4B",
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        )
        pipeline.to.assert_called_once_with("mps")
        self.assertIs(backend.pipeline, pipeline)

    def test_flux2_klein_kv_loader_uses_kv_pipeline(self):
        import torch

        pipeline = Mock()
        pipeline.to.return_value = pipeline
        with (
            patch("inference.editing.device", return_value="cuda"),
            patch(
                "diffusers.Flux2KleinKVPipeline.from_pretrained",
                return_value=pipeline,
            ) as load_pipeline,
        ):
            backend = _load_flux2_klein_image_edit_backend(
                "black-forest-labs/FLUX.2-klein-9b-kv", "flux2-klein-9b-kv"
            )

        load_pipeline.assert_called_once_with(
            "black-forest-labs/FLUX.2-klein-9b-kv",
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        )
        pipeline.to.assert_called_once_with("cuda")
        self.assertIs(backend.pipeline, pipeline)

    def test_flux2_klein_edit_uses_image_conditioned_four_step_inference(self):
        from inference.editing import Flux2KleinImageEditBackend

        source = Image.new("RGB", (16, 16), "blue")
        generated = Image.new("RGB", source.size, "red")
        pipeline = Mock()
        pipeline.return_value.images = [generated]
        backend = Flux2KleinImageEditBackend(pipeline)

        with patch("inference.editing.device", return_value="cpu"):
            result = backend.edit(source, "add a hat")

        kwargs = pipeline.call_args.kwargs
        self.assertEqual(kwargs["image"], source)
        self.assertEqual(kwargs["prompt"], "add a hat")
        self.assertEqual(kwargs["num_inference_steps"], 4)
        self.assertEqual(kwargs["guidance_scale"], 1.0)
        self.assertEqual(result.getpixel((0, 0)), (255, 0, 0))

    def test_flux2_klein_kv_edit_uses_supported_pipeline_arguments(self):
        from inference.editing import Flux2KleinImageEditBackend

        source = Image.new("RGB", (16, 16), "blue")
        generated = Image.new("RGB", source.size, "red")
        pipeline = Mock()
        pipeline.return_value.images = [generated]
        backend = Flux2KleinImageEditBackend(pipeline, guidance_scale=None)

        with patch("inference.editing.device", return_value="cpu"):
            result = backend.edit(source, "add a hat")

        kwargs = pipeline.call_args.kwargs
        self.assertEqual(kwargs["num_inference_steps"], 4)
        self.assertNotIn("guidance_scale", kwargs)
        self.assertEqual(result.getpixel((0, 0)), (255, 0, 0))

if __name__ == "__main__":
    unittest.main()
