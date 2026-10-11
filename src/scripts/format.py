"""Format Python files with isort followed by Black."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Format Python source with isort and Black."
    )
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        help="Files or directories to format (defaults to the src directory).",
    )
    args = parser.parse_args()
    targets = [str(path if path.is_absolute() else ROOT / path) for path in args.paths]
    if not targets:
        targets = [str(ROOT / "src")]

    for formatter in ("isort", "black"):
        subprocess.run(
            [sys.executable, "-m", formatter, *targets],
            cwd=ROOT,
            check=True,
        )


if __name__ == "__main__":
    main()
