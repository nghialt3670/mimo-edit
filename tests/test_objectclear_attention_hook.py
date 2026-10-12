from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from diffusers.models import attention_processor

from framework.inference.external.objectclear.pipelines.pipeline_objectclear import (
    ObjectClearPipeline,
)


class FakeAttention:
    def __init__(self) -> None:
        self.processor = FakeAttnProcessor2_0()

    def get_attention_scores(self, query, key, attention_mask=None):
        return "original attention scores"

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
            self.assertEqual(
                attention.get_attention_scores(None, None),
                "original attention scores",
            )

            _, original_state = ObjectClearPipeline.unet_store_cross_attention_scores(
                module, unet, attention_scores
            )
            self.assertTrue(unet.transformer.hooks[0].removed)
            self.assertEqual(
                attention.get_attention_scores(None, None),
                "original attention scores",
            )
            unet.transformer.hook(
                unet.transformer,
                (torch.zeros((1, 4, 24, 30)),),
                {},
            )
            self.assertEqual(module._objectclear_attention_grid, (24, 30))
            self.assertEqual(
                attention_scores[
                    "down_blocks.1.attentions.0.transformer_blocks.0.attn2"
                ],
                "original attention scores",
            )

            ObjectClearPipeline.unet_restore_attention_processor(
                module, unet, original_state
            )

        self.assertIsInstance(attention.processor, FakeAttnProcessor2_0)
        self.assertEqual(
            attention.get_attention_scores(None, None), "original attention scores"
        )
        self.assertFalse(
            hasattr(attention, "_objectclear_original_get_attention_scores")
        )
        self.assertFalse(hasattr(attention, "_objectclear_original_processor"))
        self.assertTrue(unet.transformer.hook is not None)
        self.assertTrue(unet.transformer.with_kwargs)
        self.assertFalse(hasattr(unet.transformer, "_objectclear_spatial_hook"))
