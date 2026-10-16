from __future__ import annotations

import sys
from pathlib import Path
from typing import Any


def load_viescore_scorer(viescore_root: Path, key_path: Path) -> Any:
    if not (viescore_root / "viescore" / "__init__.py").is_file():
        raise FileNotFoundError(
            f"VIESCORE_ROOT must contain viescore/__init__.py: {viescore_root}"
        )
    if not key_path.is_file():
        raise FileNotFoundError(
            "VIESCORE_KEY_PATH must point to a file containing the OpenAI API key: "
            f"{key_path}"
        )
    for import_path in (viescore_root, viescore_root / "viescore"):
        import_path_text = str(import_path.resolve())
        if import_path_text not in sys.path:
            sys.path.insert(0, import_path_text)
    try:
        from viescore import VIEScore
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Could not import VIEScore. Install its GEdit-Bench dependencies "
            "(including regex and requests) in this Python environment."
        ) from exc

    scorer = VIEScore(backbone="gpt4o", task="tie", key_path=str(key_path))
    if not hasattr(scorer.model, "model_name"):
        raise RuntimeError(
            "This VIEScore version does not expose its API model name; "
            "cannot guarantee GPT-4o evaluation."
        )
    scorer.model.model_name = "gpt-4o"
    return scorer
