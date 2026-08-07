from __future__ import annotations

import inspect
import pickle
from collections.abc import Mapping

import numpy as np
import pytest

import pipls.datasets
from pipls.datasets import PiPLSDataset, load_pulp

FEATURE_NAMES = (
    "Shives",
    "Fines B",
    "L (arith)",
    "L (lw)",
    "L (llw)",
    "W (arith)",
    "W (lw)",
    "W (llw)",
    "C (arith)",
    "C (lw)",
    "C (llw)",
    "F (arith)",
    "F (lw)",
    "F (llw)",
)
TARGET_NAMES = (
    "CSF",
    "Density",
    "TI",
    "Elongation",
    "TEA",
    "TSI",
    "Tear index",
    "s",
)


def test_load_pulp_has_linnerud_style_return_contract() -> None:
    signature = inspect.signature(load_pulp)

    assert tuple(signature.parameters) == ("return_X_y",)
    assert signature.parameters["return_X_y"].kind is inspect.Parameter.KEYWORD_ONLY
    assert signature.parameters["return_X_y"].default is False
    assert "load_pulp" in pipls.datasets.__all__

    with pytest.raises(TypeError, match="return_X_y must be a boolean"):
        load_pulp(return_X_y=1)  # type: ignore[arg-type]


def test_load_pulp_returns_labeled_immutable_dataset() -> None:
    dataset = load_pulp()

    assert isinstance(dataset, PiPLSDataset)
    assert dataset.X.shape == (46, 14)
    assert dataset.Y.shape == (46, 8)
    assert dataset.X.dtype == np.float64
    assert dataset.Y.dtype == np.float64
    assert dataset.feature_names == FEATURE_NAMES
    assert dataset.target_names == TARGET_NAMES
    assert dataset.sample_ids == tuple(f"pulp-{index:02d}" for index in range(1, 47))
    assert not dataset.X.flags.writeable
    assert not dataset.Y.flags.writeable
    assert np.isfinite(dataset.X).all()
    assert np.isfinite(dataset.Y).all()

    assert dataset.provenance["source"].endswith("10.1016/j.compchemeng.2025.109143")
    assert dataset.provenance["license"] == "CC-BY-4.0"
    assert "Lindström" in dataset.provenance["citation"]
    assert dataset.provenance["version"] == "1"

    assert isinstance(dataset.metadata, Mapping)
    assert dataset.metadata["schema_version"] == 1
    assert dataset.metadata["feature_names"] == FEATURE_NAMES
    assert dataset.metadata["target_names"] == TARGET_NAMES
    assert dataset.metadata["dimensions"] == {
        "n_samples": 46,
        "n_features": 14,
        "n_targets": 8,
    }
    with pytest.raises(TypeError):
        dataset.metadata["new"] = "value"  # type: ignore[index]


def test_load_pulp_return_X_y_matches_default_result_and_is_fresh() -> None:
    dataset = load_pulp()
    X, Y = load_pulp(return_X_y=True)
    second_X, second_Y = load_pulp(return_X_y=True)

    np.testing.assert_array_equal(X, dataset.X)
    np.testing.assert_array_equal(Y, dataset.Y)
    assert not X.flags.writeable
    assert not Y.flags.writeable
    assert not np.shares_memory(X, second_X)
    assert not np.shares_memory(Y, second_Y)

    with pytest.raises(ValueError):
        X[0, 0] = 0.0
    with pytest.raises(ValueError):
        Y[0, 0] = 0.0



def test_load_pulp_result_is_pickleable() -> None:
    dataset = load_pulp()
    restored = pickle.loads(pickle.dumps(dataset))

    assert isinstance(restored, PiPLSDataset)
    np.testing.assert_array_equal(restored.X, dataset.X)
    np.testing.assert_array_equal(restored.Y, dataset.Y)
    assert restored.feature_names == dataset.feature_names
    assert restored.target_names == dataset.target_names
    assert restored.sample_ids == dataset.sample_ids
    assert restored.provenance == dataset.provenance
    assert restored.metadata == dataset.metadata
    assert not restored.X.flags.writeable
    assert not restored.Y.flags.writeable
