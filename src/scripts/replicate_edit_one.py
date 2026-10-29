#!/usr/bin/env python3
"""Run one image edit on a Replicate model and save the result as PNG.

Usage:
    uv run src/scripts/replicate_edit_one.py --model flux_kontext \
        --image input.png --prompt "make the car red" --output out.png

Requires REPLICATE_API_TOKEN.
"""

from __future__ import annotations

import argparse
import io
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import requests
from PIL import Image

MODELS = {
    # Community models: pin the published Replicate version for reproducibility.
    "step1x": {
        "slug": "zsxkib/step1x-edit",
        "version": "12b5a5a61e3419f792eb56cfc16eed046252740ebf5d470228f9b4cf2c861610",
    },
    "bagel": {
        "slug": "bytedance/bagel",
        "version": "7dd8def79e503990740db4704fa81af995d440fefe714958531d7044d2757c9c",
    },
    # Official endpoints do not accept pinning a community version in POST.
    "flux_kontext": {"slug": "black-forest-labs/flux-kontext-dev", "version": None},
    "qwen_image_edit": {"slug": "qwen/qwen-image-edit", "version": None},
}
TERMINAL_STATUSES = ("succeeded", "failed", "canceled")


def model_input(model: str, prompt: str, input_file: Any, seed: int) -> dict:
    """Input argument names differ per endpoint."""
    if model == "step1x":
        return {
            "image": input_file,
            "prompt": prompt,
            "seed": seed,
            "size_level": 512,
            "output_format": "png",
        }
    if model == "bagel":
        return {
            "task": "image-editing",
            "image": input_file,
            "prompt": prompt,
            "seed": seed,
            "enable_thinking": False,
            "num_inference_steps": 50,
            "cfg_img_scale": 2,
            "cfg_text_scale": 4,
            "cfg_renorm_min": 1,
            "cfg_renorm_type": "text_channel",
            "timestep_shift": 3,
            "output_format": "png",
        }
    if model == "flux_kontext":
        return {
            "input_image": input_file,
            "prompt": prompt,
            "seed": seed,
            "go_fast": False,
            "aspect_ratio": "match_input_image",
            "guidance": 2.5,
            "num_inference_steps": 28,
            "output_format": "png",
        }
    if model == "qwen_image_edit":
        return {
            "image": input_file,
            "prompt": prompt,
            "seed": seed,
            "go_fast": False,
            "aspect_ratio": "match_input_image",
            "output_format": "png",
        }
    raise KeyError(f"Unknown model: {model}")


def create_prediction(client: Any, model: str, kwargs: dict):
    cfg = MODELS[model]
    if cfg["version"]:
        return client.predictions.create(
            version=f"{cfg['slug']}:{cfg['version']}", input=kwargs
        )
    return client.predictions.create(model=cfg["slug"], input=kwargs)


def output_image_url(output: Any) -> str:
    """Supports URL strings, lists, BAGEL-style dicts, and FileOutput objects."""
    if isinstance(output, str) and output.startswith(("https://", "http://")):
        return output
    if isinstance(output, (list, tuple)) and output:
        return output_image_url(output[0])
    if isinstance(output, dict):
        for key in ("image", "images", "output"):
            if key in output:
                return output_image_url(output[key])
    url = getattr(output, "url", None)
    if callable(url):
        url = url()
    if isinstance(url, str) and url.startswith(("https://", "http://")):
        return url
    raise ValueError(f"Cannot find image URL in prediction output: {str(output)[:500]}")


def default_output(model: str) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path(".out/replicate_one") / f"{model}_{stamp}.png"


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=list(MODELS))
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    parser.add_argument("--max-wait", type=float, default=2400.0)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = make_parser().parse_args(argv)
    if not args.image.is_file():
        raise SystemExit(f"Input image not found: {args.image}")
    if args.poll_seconds <= 0 or args.max_wait <= 0:
        raise SystemExit("--poll-seconds and --max-wait must be positive.")
    token = os.getenv("REPLICATE_API_TOKEN")
    if not token:
        raise SystemExit("Set REPLICATE_API_TOKEN before running.")

    import replicate

    client = replicate.Client(api_token=token)
    # The file must stay open until create() uploads it.
    with args.image.open("rb") as image_file:
        prediction = create_prediction(
            client, args.model, model_input(args.model, args.prompt, image_file, args.seed)
        )
    print(f"Created prediction {prediction.id}, waiting...", flush=True)

    started = time.monotonic()
    while True:
        prediction = client.predictions.get(prediction.id)
        if prediction.status in TERMINAL_STATUSES:
            break
        if time.monotonic() - started > args.max_wait:
            raise TimeoutError(
                f"Prediction {prediction.id} did not finish within {args.max_wait}s; "
                "it was not canceled."
            )
        time.sleep(args.poll_seconds)

    if prediction.status != "succeeded":
        raise RuntimeError(f"Prediction {prediction.status}: {prediction.error}")

    url = output_image_url(prediction.output)
    response = requests.get(url, timeout=(20, 120))
    response.raise_for_status()
    output = args.output or default_output(args.model)
    output.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(io.BytesIO(response.content)) as image:
        image.convert("RGB").save(output, format="PNG")
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
