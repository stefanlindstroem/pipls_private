"""Packaged data, validated containers, and deterministic synthetic data."""

from __future__ import annotations

from ._dataset_resources import load_pulp, load_sugarcane, load_tobacco
from ._dataset_types import PiPLSDataset, SyntheticDataTruth, _FrozenMapping
from ._synthetic_data import make_synthetic_data

__all__ = [
    "PiPLSDataset",
    "SyntheticDataTruth",
    "load_pulp",
    "load_sugarcane",
    "load_tobacco",
    "make_synthetic_data",
]


def _preserve_public_module_identity(*objects: object) -> None:
    """Keep introspection and pickle lookup on the public façade."""

    for public_object in objects:
        public_object.__module__ = __name__


_preserve_public_module_identity(
    PiPLSDataset,
    SyntheticDataTruth,
    load_pulp,
    load_sugarcane,
    load_tobacco,
    make_synthetic_data,
    _FrozenMapping,
)
del _preserve_public_module_identity
