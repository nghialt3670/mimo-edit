from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from diffusers.models import attention_processor
from diffusers.models.attention_processor import Attention, SlicedAttnProcessor

from framework.inference.external.objectclear.pipelines.pipeline_objectclear import (
    ObjectClearPipeline,
)


class FakeAttention:
    def __init__(self, heads: int = 5) -> None:
        self.processor = FakeAttnProcessor2_0()
        self.heads = heads

    def get_attention_scores(self, query, key, attention_mask=None):
        return query

    def set_processor(self, processor) -> None:
        self.processor = processor


class FakeAttnProcessor:
    pass


class FakeAttnProcessor2_0:
    pass


class FakeHook:
    def __init__(self) -> None:
        self.removed = False

    def remove(self) -> None:
        self.removed = True


class FakeTransformer:
    def __init__(self) -> None:
        self.hooks = []

    def register_forward_pre_hook(self, hook, with_kwargs=False):
        self.hook = hook
        self.with_kwargs = with_kwargs
        handle = FakeHook()
        self.hooks.append(handle)
        return handle


class FakeUnet:
    def __init__(self, attention: FakeAttention) -> None:
        self.attention = attention
        self.transformer = FakeTransformer()

    def named_modules(self):
        return [
            ("down_blocks.1.attentions.0", self.transformer),
            ("down_blocks.1.attentions.0.transformer_blocks.0.attn2", self.attention)
        ]


class ObjectClearAttentionHookTests(unittest.TestCase):
    def test_attention_grid_resizes_non_square_attention_to_mask(self) -> None:
        attention = torch.ones((4, 720, 7))
        mask = torch.ones((2, 1, 48, 48))

        result = ObjectClearPipeline.resize_attn_map_divide2(
            SimpleNamespace(),
            attention,
            mask,
            fuse_index=5,
            attention_grid=(24, 30),
            attention_heads=2,
        )

        self.assertEqual(tuple(result.shape), (2, 1, 48, 48))

    def test_single_attention_batch_expands_to_cfg_mask_batch(self) -> None:
        attention = torch.ones((5, 576, 7))
        mask = torch.ones((2, 1, 48, 48))

        result = ObjectClearPipeline.resize_attn_map_divide2(
            SimpleNamespace(),
            attention,
            mask,
            fuse_index=5,
            attention_grid=(24, 24),
            attention_heads=5,
        )

        self.assertEqual(tuple(result.shape), (2, 1, 48, 48))

    def test_hook_can_be_installed_again_after_interrupted_call(self) -> None:
        attention = FakeAttention()
        unet = FakeUnet(attention)
        attention_scores = {}
        module = SimpleNamespace()

        with (
            patch.object(attention_processor, "Attention", FakeAttention),
            patch.object(attention_processor, "AttnProcessor", FakeAttnProcessor),
            patch.object(attention_processor, "AttnProcessor2_0", FakeAttnProcessor2_0),
        ):
            _, _ = ObjectClearPipeline.unet_store_cross_attention_scores(
                module, unet, attention_scores
            )
            query = torch.ones((1, 2, 3))
            self.assertIs(attention.get_attention_scores(query, None), query)

            _, original_state = ObjectClearPipeline.unet_store_cross_attention_scores(
                module, unet, attention_scores
            )
            self.assertTrue(unet.transformer.hooks[0].removed)
            self.assertIs(attention.get_attention_scores(query, None), query)
            unet.transformer.hook(
                unet.transformer,
                (torch.zeros((1, 4, 24, 30)),),
                {},
            )
            self.assertEqual(module._objectclear_attention_grid, (24, 30))
            self.assertTrue(
                torch.equal(
                    attention_scores[
                        "down_blocks.1.attentions.0.transformer_blocks.0.attn2"
                    ],
                    query,
                )
            )

            ObjectClearPipeline.unet_restore_attention_processor(
                module, unet, original_state
            )
        self.assertIsInstance(attention.processor, FakeAttnProcessor2_0)
        query = torch.ones((1, 2, 3))
        self.assertIs(attention.get_attention_scores(query, None), query)
        self.assertFalse(
            hasattr(attention, "_objectclear_original_get_attention_scores")
        )
        self.assertFalse(hasattr(attention, "_objectclear_original_processor"))
        self.assertTrue(unet.transformer.hook is not None)
        self.assertTrue(unet.transformer.with_kwargs)
        self.assertFalse(hasattr(unet.transformer, "_objectclear_spatial_hook"))

    def test_attention_capture_reassembles_cuda_slicing_chunks(self) -> None:
        layer_name = "down_blocks.1.attentions.0.transformer_blocks.0.attn2"
        attention = FakeAttention(heads=10)
        unet = FakeUnet(attention)
        attention_scores = {layer_name: torch.zeros((3, 720, 77))}
        module = SimpleNamespace()

        with (
            patch.object(attention_processor, "Attention", FakeAttention),
            patch.object(attention_processor, "AttnProcessor", FakeAttnProcessor),
            patch.object(attention_processor, "AttnProcessor2_0", FakeAttnProcessor2_0),
        ):
            _, original_state = ObjectClearPipeline.unet_store_cross_attention_scores(
                module, unet, attention_scores
            )
            self.assertEqual(attention_scores, {})

            for chunk_index in range(4):
                query = torch.full((5, 720, 77), float(chunk_index))
                attention.get_attention_scores(query, None)

            captured = attention_scores[layer_name]
            self.assertEqual(tuple(captured.shape), (20, 720, 77))
            for chunk_index in range(4):
                self.assertTrue(torch.all(captured[chunk_index * 5] == chunk_index))

            result = ObjectClearPipeline.resize_attn_map_divide2(
                module,
                captured,
                torch.ones((2, 1, 48, 48)),
                fuse_index=5,
                attention_grid=(24, 30),
                attention_heads=10,
            )
            self.assertEqual(tuple(result.shape), (2, 1, 48, 48))
            ObjectClearPipeline.unet_restore_attention_processor(
                module, unet, original_state
            )

    def test_diffusers_sliced_processor_capture_has_full_cfg_batch(self) -> None:
        layer_name = "down_blocks.1.attentions.0.transformer_blocks.0.attn2"
        attention = Attention(
            query_dim=40,
            cross_attention_dim=40,
            heads=10,
            dim_head=4,
            processor=SlicedAttnProcessor(slice_size=5),
        )
        unet = FakeUnet(attention)
        attention_scores = {}
        module = SimpleNamespace()

        _, original_state = ObjectClearPipeline.unet_store_cross_attention_scores(
            module, unet, attention_scores
        )
        hidden_states = torch.randn((2, 24 * 30, 40))
        encoder_hidden_states = torch.randn((2, 77, 40))
        attention(hidden_states, encoder_hidden_states=encoder_hidden_states)

        captured = attention_scores[layer_name]
        self.assertEqual(tuple(captured.shape), (20, 720, 77))
        result = ObjectClearPipeline.resize_attn_map_divide2(
            module,
            captured,
            torch.ones((2, 1, 48, 60)),
            fuse_index=5,
            attention_grid=(24, 30),
            attention_heads=10,
        )
        self.assertEqual(tuple(result.shape), (2, 1, 48, 60))
        ObjectClearPipeline.unet_restore_attention_processor(
            module, unet, original_state
        )
