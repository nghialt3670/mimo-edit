from __future__ import annotations

import re

from PIL import Image

from .pool import ModelPool

_PROMPT = (
    "Locate the region described by: {query}. "
    'Answer only with JSON in the form {{"bbox_2d": [x1, y1, x2, y2]}}, '
    "with coordinates normalized to 0-1000."
)
_JSON_PREFIX = '{"bbox_2d": ['
_NUMBER = re.compile(r"-?\d+(?:\.\d+)?")
_NORMALIZED_MAX = 1000.0


def qwen_locate_region(
    pool: ModelPool,
    image: Image.Image,
    prompt: str,
) -> list[float]:
    """Return a pixel box [x1, y1, x2, y2] for the prompt, or the full image if unparsable."""
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("prompt must be a non-empty string")

    source = image.convert("RGB")
    box = _generate_normalized_box(pool, source, prompt.strip())
    if box is None:
        return [0.0, 0.0, float(source.width), float(source.height)]
    return _to_pixel_box(box, source.width, source.height)


def _generate_normalized_box(
    pool: ModelPool,
    image: Image.Image,
    query: str,
) -> list[float] | None:
    import torch

    pipelines = pool.qwen_locate_pipelines
    processor = pipelines.processor
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": _PROMPT.format(query=query)},
            ],
        }
    ]
    text = (
        processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        + _JSON_PREFIX
    )
    inputs = processor(text=[text], images=[image], return_tensors="pt").to(pool.device)
    with torch.inference_mode():
        output_ids = pipelines.model.generate(
            **inputs,
            max_new_tokens=pool.config.qwen_locate.max_new_tokens,
            do_sample=False,
        )
    generated = processor.batch_decode(
        output_ids[:, inputs.input_ids.shape[1] :],
        skip_special_tokens=True,
    )[0]
    numbers = _NUMBER.findall(generated)
    if len(numbers) < 4:
        return None
    return [float(value) for value in numbers[:4]]


def _to_pixel_box(box: list[float], width: int, height: int) -> list[float]:
    x1, y1, x2, y2 = box
    x1, x2 = sorted((x1, x2))
    y1, y2 = sorted((y1, y2))
    scale_x = width / _NORMALIZED_MAX
    scale_y = height / _NORMALIZED_MAX
    x1 = min(max(x1 * scale_x, 0.0), float(width))
    x2 = min(max(x2 * scale_x, 0.0), float(width))
    y1 = min(max(y1 * scale_y, 0.0), float(height))
    y2 = min(max(y2 * scale_y, 0.0), float(height))
    if x2 <= x1 or y2 <= y1:
        return [0.0, 0.0, float(width), float(height)]
    return [x1, y1, x2, y2]


__all__ = ["qwen_locate_region"]
