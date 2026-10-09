from __future__ import annotations

from pathlib import Path


def find_project_root(start: Path | None = None) -> Path:
    """Find the repository root from the current directory or this source file."""
    search_starts = [(start or Path.cwd()).resolve()]
    source_root = Path(__file__).resolve().parents[3]
    if source_root not in search_starts:
        search_starts.append(source_root)

    for current in search_starts:
        for candidate in (current, *current.parents):
            if (candidate / "pyproject.toml").is_file():
                return candidate
    raise FileNotFoundError(
        "Could not find pyproject.toml from the current directory or installed source."
    )


def load_project_environment(root: Path | None = None) -> Path:
    """Load the repository .env without overriding already-set variables."""
    from dotenv import load_dotenv

    project_root = (root or find_project_root()).resolve()
    load_dotenv(project_root / ".env")
    return project_root
