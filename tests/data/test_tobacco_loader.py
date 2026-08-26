from __future__ import annotations

import inspect
import pickle
from collections.abc import Mapping

import numpy as np
import pytest

import pipls.datasets
from pipls.datasets import PiPLSDataset, load_tobacco

TARGET_NAMES = (
    "Total Alkaloids",
    "Reducing Sugars",
    "Total Sugars",
    "Total Nitrogen",
    "K",
    "cl",
    "pH",
    "Starch",
    "Neochlorogenic Acid",
    "Chlorogenic Acid",
    "Cryptochlorogenic Acid",
    "Scopoletin",
    "Rutin",
)


def test_load_tobacco_has_named_loader_return_contract() -> None:
    signature = inspect.signature(load_tobacco)

    assert tuple(signature.parameters) == ("return_X_y",)
    assert signature.parameters["return_X_y"].kind is inspect.Parameter.KEYWORD_ONLY
    assert signature.parameters["return_X_y"].default is False
    assert "load_tobacco" in pipls.datasets.__all__

    with pytest.raises(TypeError, match="return_X_y must be a boolean"):
        load_tobacco(return_X_y=1)  # type: ignore[arg-type]


def test_load_tobacco_returns_labeled_immutable_dataset() -> None:
    dataset = load_tobacco()
    wavenumbers = np.asarray(dataset.feature_names, dtype=np.float64)

    assert isinstance(dataset, PiPLSDataset)
    assert dataset.X.shape == (347, 1557)
    assert dataset.Y.shape == (347, 13)
    assert dataset.X.dtype == np.float64
    assert dataset.Y.dtype == np.float64
    assert dataset.feature_names[0] == "10001.0283203125"
    assert dataset.feature_names[-1] == "3999.63989257813"
    assert np.all(np.diff(wavenumbers) < 0.0)
    assert dataset.target_names == TARGET_NAMES
    assert not dataset.X.flags.writeable
    assert not dataset.Y.flags.writeable
    assert np.isfinite(dataset.X).all()
    assert np.isfinite(dataset.Y).all()

    provenance = dataset.metadata["provenance"]
    assert isinstance(provenance, Mapping)
    assert provenance["source"].endswith("10.17632/9z7dgdtggk.1")
    assert provenance["license"] == "CC-BY-4.0"
    assert "Chen" in provenance["citation"]
    assert provenance["version"] == "1"

    assert isinstance(dataset.metadata, Mapping)
    assert dataset.metadata["schema_version"] == 1
    assert dataset.metadata["feature_names"] == dataset.feature_names
    assert dataset.metadata["target_names"] == TARGET_NAMES
    assert dataset.metadata["dimensions"] == {
        "n_samples": 347,
        "n_features": 1557,
        "n_targets": 13,
    }
    variables = dataset.metadata["variables"]
    assert isinstance(variables, Mapping)
    predictors = variables["predictors"]
    assert isinstance(predictors, Mapping)
    axis = predictors["axis"]
    assert isinstance(axis, Mapping)
    assert axis["name"] == "wavenumber"
    assert axis["unit"] == "cm^-1"
    assert axis["ordering"] == "decreasing"
    assert axis["step"] == pytest.approx(-3.856933436849431)
    with pytest.raises(TypeError):
        dataset.metadata["new"] = "value"  # type: ignore[index]


def test_load_tobacco_return_X_y_matches_default_result_and_is_fresh() -> None:
    dataset = load_tobacco()
    second_dataset = load_tobacco()
    X, Y = load_tobacco(return_X_y=True)
    second_X, second_Y = load_tobacco(return_X_y=True)

    assert not np.shares_memory(dataset.X, second_dataset.X)
    assert not np.shares_memory(dataset.Y, second_dataset.Y)
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



def test_load_tobacco_result_is_pickleable() -> None:
    dataset = load_tobacco()
    restored = pickle.loads(pickle.dumps(dataset))

    assert isinstance(restored, PiPLSDataset)
    np.testing.assert_array_equal(restored.X, dataset.X)
    np.testing.assert_array_equal(restored.Y, dataset.Y)
    assert restored.feature_names == dataset.feature_names
    assert restored.target_names == dataset.target_names
    assert restored.metadata == dataset.metadata
    assert not restored.X.flags.writeable
    assert not restored.Y.flags.writeable
