from __future__ import annotations

import gc

from .editing import (
    FLUX2_KLEIN_MODEL_IDS,
    _load_flux2_klein_image_edit_backend,
)
from .inpainting import (
    DEFAULT_OBJECT_CLEAR_MODEL_ID,
    _load_object_clear,
)
from .segmentation import DEFAULT_SEGMENTATION_MODEL_ID, _load_sam3


def warmup_default_models() -> list[str]:
    """Download and load each default model once, releasing it before the next."""
    model_loaders = [
        (
            DEFAULT_SEGMENTATION_MODEL_ID,
            lambda: _load_sam3(DEFAULT_SEGMENTATION_MODEL_ID),
        ),
        (
            FLUX2_KLEIN_MODEL_IDS["flux2-klein-4b"],
            lambda: _load_flux2_klein_image_edit_backend(
                FLUX2_KLEIN_MODEL_IDS["flux2-klein-4b"],
                "flux2-klein-4b",
            ),
        ),
        (
            DEFAULT_OBJECT_CLEAR_MODEL_ID,
            lambda: _load_object_clear(DEFAULT_OBJECT_CLEAR_MODEL_ID),
        ),
    ]
    warmed_models = []
    for model_id, loader in model_loaders:
        print(f"Loading {model_id}...", flush=True)
        pipeline = loader()
        del pipeline
        gc.collect()

        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif torch.backends.mps.is_available():
            torch.mps.empty_cache()
        warmed_models.append(model_id)
        print(f"Ready: {model_id}", flush=True)
    return warmed_models
