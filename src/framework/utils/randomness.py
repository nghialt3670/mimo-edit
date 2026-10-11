from __future__ import annotations

import os
import random

_MAX_SEED = 2**32 - 1
_initialized_seed: int | None = None


def get_random_seed() -> int:
    """Read the required project-wide seed from the process environment."""
    from .project import load_project_environment

    load_project_environment()
    raw_seed = os.environ.get("RANDOM_SEED")
    if raw_seed is None:
        raise ValueError(
            "RANDOM_SEED is required; set it in the project .env or process environment."
        )
    try:
        seed = int(raw_seed)
    except ValueError as exc:
        raise ValueError(
            f"RANDOM_SEED must be an integer between 0 and {_MAX_SEED}."
        ) from exc
    if not 0 <= seed <= _MAX_SEED:
        raise ValueError(f"RANDOM_SEED must be between 0 and {_MAX_SEED}.")
    return seed


def seed_random_generators() -> int:
    """Seed Python, NumPy, and Torch RNGs from the required project seed."""
    global _initialized_seed

    seed = get_random_seed()
    if seed == _initialized_seed:
        return seed

    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    _initialized_seed = seed
    return seed
