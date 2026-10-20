"""FLUX.2 Klein 4B editing, insertion, and removal functions."""

from .add_object import (
    flux2_klein_add_object_with_box,
    flux2_klein_add_object_with_mask,
)
from .edit_object import flux2_klein_edit_object
from .remove_object import flux2_klein_remove_object
from .replace_object import flux2_klein_replace_object

__all__ = [
    "flux2_klein_add_object_with_box",
    "flux2_klein_add_object_with_mask",
    "flux2_klein_edit_object",
    "flux2_klein_remove_object",
    "flux2_klein_replace_object",
]
