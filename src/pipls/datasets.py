"""Packaged data, validated containers, and deterministic Π-PLS generators."""

from __future__ import annotations

from ._dataset_resources import load_pulp, load_sugarcane, load_tobacco
from ._dataset_types import (
    PiPLSDataset,
    PiPLSRegressionTruth,
    SyntheticDataTruth,
    _FrozenMapping,
)
from ._synthetic_data import (
    make_pipls_regression,
    make_pipls_train_test,
    make_synthetic_data,
)

__all__ = [
    "PiPLSDataset",
    "SyntheticDataTruth",
    "PiPLSRegressionTruth",
    "load_pulp",
    "load_sugarcane",
    "load_tobacco",
    "make_synthetic_data",
    "make_pipls_regression",
    "make_pipls_train_test",
]


def _preserve_public_module_identity(*objects: object) -> None:
    """Keep introspection and pickle lookup on the public façade."""

    for public_object in objects:
        public_object.__module__ = __name__


_preserve_public_module_identity(
    PiPLSDataset,
    SyntheticDataTruth,
    PiPLSRegressionTruth,
    load_pulp,
    load_sugarcane,
    load_tobacco,
    make_synthetic_data,
    make_pipls_regression,
    make_pipls_train_test,
    _FrozenMapping,
)
del _preserve_public_module_identity
