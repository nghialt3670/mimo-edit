"""Cut the edited object out of a ``test_edit_object`` debug folder.

Uses ``after_harmonize.png`` as colour and ``matte_raw.png`` as alpha, and
writes ``cutout.png`` (RGBA, transparent background) next to them. The matte is
used as is, so fur keeps its fractional alpha.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def make_cutout(folder: Path, crop_to_object: bool = False) -> Path:
    with Image.open(folder / "after_harmonize.png") as image:
        rgb = image.convert("RGB")
    with Image.open(folder / "matte_raw.png") as matte:
        alpha = matte.convert("L")
    if alpha.size != rgb.size:
        raise ValueError(f"matte {alpha.size} and image {rgb.size} differ in size")
    cutout = rgb.convert("RGBA")
    cutout.putalpha(alpha)
    if crop_to_object:
        box = Image.fromarray((np.asarray(alpha) > 0).astype(np.uint8) * 255).getbbox()
        if box is not None:
            cutout = cutout.crop(box)
    path = folder / ("cutout_cropped.png" if crop_to_object else "cutout.png")
    cutout.save(path)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path, help="a *_debug folder")
    parser.add_argument(
        "--crop", action="store_true", help="also save a tight crop of the object"
    )
    args = parser.parse_args()
    print(make_cutout(args.folder))
    if args.crop:
        print(make_cutout(args.folder, crop_to_object=True))


if __name__ == "__main__":
    main()
