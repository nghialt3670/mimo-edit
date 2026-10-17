"""Create a simple ML-generated 3D shape proxy inside an exact image box.

Uses OpenAI Point-E `base40M-textvec` (1024-point, text-conditioned model).
No hard-coded object categories, shapes, meshes, or downloaded asset library.

Installation (with an existing uv Python environment containing PyTorch):

    uv add pillow numpy scipy scikit-image requests filelock tqdm ftfy regex
    uv add 'git+https://github.com/openai/CLIP.git'
    uv add 'git+https://github.com/openai/point-e.git'

Usage:

    from shape_proxy_ml import add_shape_proxy

    result = add_shape_proxy('original.png', (200, 150, 450, 500), 'wooden chair')
    result.save('shape_proposal.png')

The model generates geometry solely from text (NOT from the photograph). The
original image is only used as a background, and the requested box is enforced
in the compositor. Perspective/pose are NOT inferred from the scene; adjust
azimuth and elevation to match it as well as possible.

The first call downloads Point-E and OpenAI CLIP ViT-L/14 model weights. Point-E's
40M parameter count excludes that text encoder, so this is lightweight relative
to modern diffusion editors, not a tiny on-disk installation.
"""

from __future__ import annotations

import argparse
import math
from functools import lru_cache
from pathlib import Path
from typing import Literal

import numpy as np
from PIL import Image, ImageColor, ImageDraw, ImageFilter


def _choose_device(device: str) -> str:
    if device != "auto":
        return device
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _patch_point_e_for_mps() -> None:
    """Work around Point-E's float64 -> MPS error (openai/point-e#78).

    Upstream _extract_into_tensor() converts a NumPy float64 schedule to an
    MPS tensor *before* casting it to float32. Apple MPS rejects float64.
    This patch casts on the CPU first, and leaves CPU/CUDA unchanged.
    """
    import torch
    from point_e.diffusion import gaussian_diffusion as gd

    if getattr(gd, "_shape_proxy_mps_patched", False):
        return

    original = gd._extract_into_tensor

    def extract_with_mps_float32(arr, timesteps, broadcast_shape):
        if timesteps.device.type != "mps":
            return original(arr, timesteps, broadcast_shape)

        # Cast NumPy coefficients BEFORE moving them to MPS.
        coefficients = np.asarray(arr, dtype=np.float32)
        result = torch.from_numpy(coefficients).to(timesteps.device)[timesteps]
        while result.ndim < len(broadcast_shape):
            result = result[..., None]
        return result + torch.zeros(
            broadcast_shape, dtype=torch.float32, device=timesteps.device
        )

    gd._extract_into_tensor = extract_with_mps_float32
    gd._shape_proxy_mps_patched = True


@lru_cache(maxsize=3)
def _get_sampler(device_str: str, steps: int):
    """Load just the 40M base model, not Point-E's optional upsampler."""
    import torch
    from point_e.diffusion.configs import DIFFUSION_CONFIGS, diffusion_from_config
    from point_e.diffusion.sampler import PointCloudSampler
    from point_e.models.configs import MODEL_CONFIGS, model_from_config
    from point_e.models.download import load_checkpoint

    name = "base40M-textvec"
    dev = torch.device(device_str)
    if dev.type == "mps":
        _patch_point_e_for_mps()
    model = model_from_config(MODEL_CONFIGS[name], dev)
    model.eval()
    model.load_state_dict(load_checkpoint(name, dev))

    # IMPORTANT: all per-stage arguments must have length 1 for a single model.
    # The upstream sampler's defaults have length 2 (base + upsampler).
    sampler = PointCloudSampler(
        device=dev,
        models=[model],
        diffusions=[diffusion_from_config(DIFFUSION_CONFIGS[name])],
        num_points=[1024],
        aux_channels=["R", "G", "B"],
        guidance_scale=[3.0],
        model_kwargs_key_filter=["texts"],
        use_karras=[True],
        karras_steps=[steps],
        sigma_min=[1e-3],
        sigma_max=[120],
        s_churn=[3],
    )
    return sampler


def _generate_points(prompt: str, device: str, steps: int, seed: int) -> np.ndarray:
    """Generate a 1024x3 text-conditioned point cloud."""
    import torch

    torch.manual_seed(seed)
    sampler = _get_sampler(device, steps)
    with torch.inference_mode():
        sample = sampler.sample_batch(
            batch_size=1,
            model_kwargs={"texts": [prompt]},
        )
        cloud = sampler.output_to_point_clouds(sample)[0]
    return np.asarray(cloud.coords, dtype=np.float32)


def _parse_color(color: str | tuple[int, int, int]) -> tuple[int, int, int]:
    """Accept color names (e.g. 'orange'), hex codes, or (R, G, B)."""
    if isinstance(color, str):
        if color.lower() == "auto":
            raise ValueError("'auto' needs the background image; use add_shape_proxy")
        try:
            return ImageColor.getrgb(color)
        except ValueError as exc:
            raise ValueError(f"Invalid color: {color!r}") from exc
    if len(color) != 3 or any(
        not isinstance(c, int) or not 0 <= c <= 255 for c in color
    ):
        raise ValueError(
            "color must be a name, hex string, or RGB tuple of 0..255 ints"
        )
    return tuple(color)


def _auto_contrast_color(region: Image.Image) -> tuple[int, int, int]:
    """Choose a visible color against the actual pixels behind the proxy."""
    pixels = np.asarray(region.convert("RGB").resize((64, 64)), dtype=np.float32)
    pixels = pixels.reshape(-1, 3)
    palette = [
        (255, 106, 55),  # orange
        (32, 218, 240),  # cyan
        (247, 58, 177),  # magenta
        (247, 224, 60),  # yellow
        (107, 85, 242),  # violet
    ]
    # Choose the color whose lower-quartile RGB distance from the background
    # is largest. This avoids using plain gray on gray and similar colors.
    return max(
        palette,
        key=lambda rgb: float(
            np.quantile(
                np.linalg.norm(pixels - np.asarray(rgb, dtype=np.float32), axis=1),
                0.25,
            )
        ),
    )


def render_points_in_box(
    points: np.ndarray,
    box_size: tuple[int, int],
    *,
    azimuth: float = 35,
    elevation: float = 20,
    style: Literal["clay", "points"] = "clay",
    fit: Literal["contain", "stretch", "cover"] = "contain",
    margin: float = 0.02,
    gray: int = 145,
    color: str | tuple[int, int, int] | None = None,
) -> Image.Image:
    """Project a learned 3D point cloud into a transparent RGBA box-sized image.

    This is deterministic placement, not a second generative shape renderer.
    Kept separate from model inference so it can be tested without weights.
    """
    w, h = box_size
    if w <= 0 or h <= 0:
        raise ValueError("box_size must be positive")
    if style not in ("clay", "points"):
        raise ValueError("style must be 'clay' or 'points'")
    if fit not in ("contain", "stretch", "cover"):
        raise ValueError("fit must be 'contain', 'stretch', or 'cover'")
    if not (0 <= margin <= 0.25):
        raise ValueError("margin must be between 0 and 0.25")

    xyz = np.asarray(points, dtype=np.float64)
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError("Expected point cloud of shape (N, 3)")
    xyz = xyz[np.isfinite(xyz).all(axis=1)]
    if xyz.shape[0] < 5:
        raise ValueError("Generated point cloud has too few valid points")

    # Remove extreme model outliers so a single bad point cannot shrink the shape.
    lo = np.percentile(xyz, 1, axis=0)
    hi = np.percentile(xyz, 99, axis=0)
    xyz = xyz[((xyz >= lo) & (xyz <= hi)).all(axis=1)]
    if xyz.shape[0] < 5:
        raise ValueError("Point cloud collapsed after outlier filtering")
    xyz = xyz - np.median(xyz, axis=0)

    # Point-E uses z as its vertical axis. Rotate about z (azimuth), then tilt.
    a, e = math.radians(azimuth), math.radians(elevation)
    xx = math.cos(a) * xyz[:, 0] - math.sin(a) * xyz[:, 1]
    depth = math.sin(a) * xyz[:, 0] + math.cos(a) * xyz[:, 1]
    yy = math.cos(e) * xyz[:, 2] - math.sin(e) * depth
    zz = math.sin(e) * xyz[:, 2] + math.cos(e) * depth

    ext_x = float(np.ptp(xx))
    ext_y = float(np.ptp(yy))
    if max(ext_x, ext_y) < 1e-8:
        raise ValueError("Point cloud collapsed to a single projected location")

    # Fit the projected shape. "contain" keeps proportions and may leave space;
    # "stretch" fills both dimensions (possibly distorting shape);
    # "cover" fills both dimensions without distortion but clips the excess.
    pad = round(min(w, h) * margin)
    radius = max(1, round(min(w, h) * (0.013 if style == "clay" else 0.005)))
    usable_w = max(1, w - 1 - 2 * (pad + radius))
    usable_h = max(1, h - 1 - 2 * (pad + radius))
    scale_x = usable_w / max(ext_x, 1e-8)
    scale_y = usable_h / max(ext_y, 1e-8)
    if fit == "contain":
        sx = sy = min(scale_x, scale_y)
    elif fit == "cover":
        sx = sy = max(scale_x, scale_y)
    else:  # stretch: exact silhouette extent in both dimensions
        sx, sy = scale_x, scale_y

    # Center in x and anchor the projected object's lowest point at bottom.
    x = (xx - (xx.min() + xx.max()) / 2) * sx + (w - 1) / 2
    y = (yy.min() - yy) * sy + h - 1 - pad - radius

    base_rgb = (gray, gray, gray) if color is None else _parse_color(color)

    # Draw far points first. For 'clay', larger disks create rough solid mass;
    # the optional blur softens gaps, but the model still supplies all geometry.
    out = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(out)
    order = np.argsort(zz)
    for i in order:
        px, py = float(x[i]), float(y[i])
        # Vary brightness slightly with depth while retaining the chosen hue.
        shade = float(np.clip(1.0 + zz[i] * 0.18, 0.70, 1.20))
        rgb = tuple(int(np.clip(c * shade, 0, 255)) for c in base_rgb)
        rr = radius
        draw.ellipse((px - rr, py - rr, px + rr, py + rr), fill=(*rgb, 255))

    if style == "clay" and min(w, h) >= 60:
        # Mild blur only inside the target box; avoids harsh dotted surfaces.
        out = out.filter(ImageFilter.GaussianBlur(radius=0.55))
    return out


def add_shape_proxy(
    image: Image.Image | str | Path,
    box: tuple[int, int, int, int],
    prompt: str,
    *,
    device: str = "auto",
    seed: int = 0,
    steps: int = 48,
    azimuth: float = 35,
    elevation: float = 20,
    style: Literal["clay", "points"] = "clay",
    fit: Literal["contain", "stretch", "cover"] = "contain",
    margin: float = 0.02,
    color: str | tuple[int, int, int] | None = None,
) -> Image.Image:
    """Return a PIL RGB image with an ML-generated shape fitted inside `box`.

    Args:
        image: PIL image or filesystem path.
        box: (left, top, right, bottom), original-image pixel coordinates.
        prompt: Simple, visually recognizable object prompt (e.g. 'wooden chair').
        device: 'auto', 'mps', 'cpu', or 'cuda'. Try 'cpu' if MPS fails.
        seed: Reproducibility seed.
        steps: Point-E denoising steps; 48 trades quality for speed (64 upstream).
        azimuth/elevation: Manually tune projected camera angle in degrees.
        style: 'clay' rough mass or 'points' sparse skeleton dots.
        fit: 'contain' preserves aspect ratio; 'stretch' fills the bbox in both
             dimensions; 'cover' fills it without stretching but clips edges.
        margin: Bbox inset fraction; use 0 to bring the silhouette to the edges.
        color: Optional proxy color, e.g. 'orange', '#ff6600', (255, 80, 20),
               or 'auto' to choose a contrasting color from the bbox background.
               Defaults to the original gray rendering.
    """
    if not prompt.strip():
        raise ValueError("prompt cannot be empty")
    if len(box) != 4:
        raise ValueError("box must be (x1, y1, x2, y2)")
    bg = (Image.open(image) if isinstance(image, (str, Path)) else image).convert("RGB")
    x1, y1, x2, y2 = map(int, box)
    if not (0 <= x1 < x2 <= bg.width and 0 <= y1 < y2 <= bg.height):
        raise ValueError(f"box {box} must be nonempty and inside image {bg.size}")

    if isinstance(color, str) and color.lower() == "auto":
        color = _auto_contrast_color(bg.crop((x1, y1, x2, y2)))

    points = _generate_points(prompt.strip(), _choose_device(device), steps, seed)
    local = render_points_in_box(
        points,
        (x2 - x1, y2 - y1),
        azimuth=azimuth,
        elevation=elevation,
        style=style,
        fit=fit,
        margin=margin,
        color=color,
    )
    output = bg.convert("RGBA")
    output.alpha_composite(local, (x1, y1))
    return output.convert("RGB")


def _cli():
    parser = argparse.ArgumentParser(description="Make a coarse Point-E object proxy")
    parser.add_argument("image", help="Source image")
    parser.add_argument("prompt", help="Object name, e.g. 'office chair'")
    parser.add_argument(
        "--box", nargs=4, type=int, required=True, metavar=("X1", "Y1", "X2", "Y2")
    )
    parser.add_argument("--output", default="proxy.png")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--steps", type=int, default=48)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--azimuth", type=float, default=35)
    parser.add_argument("--elevation", type=float, default=20)
    parser.add_argument("--style", choices=["clay", "points"], default="clay")
    parser.add_argument(
        "--fit", choices=["contain", "stretch", "cover"], default="contain"
    )
    parser.add_argument("--margin", type=float, default=0.02)
    parser.add_argument("--color", default=None, help="e.g. orange, #ff6600, or auto")
    args = parser.parse_args()
    result = add_shape_proxy(
        args.image,
        tuple(args.box),
        args.prompt,
        device=args.device,
        steps=args.steps,
        seed=args.seed,
        azimuth=args.azimuth,
        elevation=args.elevation,
        style=args.style,
        fit=args.fit,
        margin=args.margin,
        color=args.color,
    )
    result.save(args.output)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    _cli()
