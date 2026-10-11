from __future__ import annotations

from math import isfinite

import torch


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if (
        getattr(torch.backends, "mps", None) is not None
        and torch.backends.mps.is_available()
    ):
        return "mps"
    return "cpu"


def torch_dtype(
    device_name: str,
    *,
    cuda_dtype: torch.dtype | None = None,
    mps_dtype: torch.dtype | None = None,
) -> torch.dtype:
    if device_name.startswith("cuda"):
        return cuda_dtype if cuda_dtype is not None else torch.float16
    if device_name == "mps":
        return mps_dtype if mps_dtype is not None else torch.float32
    return torch.float32


def validate_positive_integer(name: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def validate_non_negative_integer(name: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")


def validate_non_negative_number(name: str, value: object) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not isfinite(value)
        or value < 0
    ):
        raise ValueError(f"{name} must be a finite non-negative number")


def validate_number_range(
    name: str,
    value: object,
    minimum: float,
    maximum: float,
    *,
    maximum_inclusive: bool = True,
) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not isfinite(value)
        or value < minimum
        or (value > maximum if maximum_inclusive else value >= maximum)
    ):
        raise ValueError(f"{name} must be between {minimum:g} and {maximum:g}")


def validate_box(values: list[float]) -> tuple[int, int, int, int]:
    if len(values) != 4:
        raise ValueError("box must be [x1, y1, x2, y2]")
    result = tuple(round(float(value)) for value in values)
    if result[2] <= result[0] or result[3] <= result[1]:
        raise ValueError("box must have positive width and height")
    return result
