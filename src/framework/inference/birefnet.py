from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as functional
from PIL import Image

from .pool import ModelPool

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@torch.inference_mode()
def birefnet_alpha_matte(pool: ModelPool, image: Image.Image) -> np.ndarray:
    """Float alpha matte in [0, 1] of the salient object, at ``image`` size.

    One BiRefNet HR-Matting pass at ``config.birefnet.resolution`` (FP16 on CUDA).
    The matte stays float all the way, so semi-transparent fur, hair and edges
    keep their fractional values; nothing is thresholded or blurred.
    """
    if not isinstance(image, Image.Image):
        raise TypeError("image must be a Pillow image")
    model = pool.birefnet_matting_model
    parameter = next(model.parameters())
    size = pool.config.birefnet.resolution

    rgb = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0)
    tensor = functional.interpolate(
        tensor, size=(size, size), mode="bicubic", align_corners=False
    ).clamp_(0, 1)
    mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
    tensor = ((tensor - mean) / std).to(parameter.device, parameter.dtype)

    matte = model(tensor)[-1].sigmoid().float()
    matte = functional.interpolate(
        matte, size=(image.height, image.width), mode="bilinear", align_corners=False
    )
    return matte[0, 0].clamp_(0, 1).cpu().numpy()
