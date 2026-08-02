from __future__ import annotations

import hashlib
import inspect
import pickle
from collections.abc import Mapping
from importlib import resources
from pathlib import Path

import numpy as np
import pytest

import pipls
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
RESOURCE_HASHES = {
    "X.csv": "b26d1639339c406bf91070e2ad8ceb8df59d3ad1f90ece7c5cd5b18264d63b45",
    "Y.csv": "b600264d2319efce3500c6d7adbf5490b62e6d42a1e32af7df62709b1975ce93",
    "LICENSE.txt": "e21d8674cbe4b64a371cd530a0ea8fa4727c1de7005dc707e86b55ea8e6f14d7",
}


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _repository_matrix(name: str) -> np.ndarray:
    return np.loadtxt(
        _repository_root() / "datasets" / "pulp" / name,
        delimiter=",",
        skiprows=1,
        dtype=np.float64,
    )


def test_load_pulp_has_linnerud_style_return_contract() -> None:
    signature = inspect.signature(load_pulp)

    assert tuple(signature.parameters) == ("return_X_y",)
    assert signature.parameters["return_X_y"].kind is inspect.Parameter.KEYWORD_ONLY
    assert signature.parameters["return_X_y"].default is False
    assert "load_pulp" in pipls.datasets.__all__
    assert not hasattr(pipls, "load_pulp")

    with pytest.raises(TypeError, match="return_X_y must be a boolean"):
        load_pulp(return_X_y=1)  # type: ignore[arg-type]


def test_load_pulp_returns_labeled_immutable_dataset() -> None:
    dataset = load_pulp()

    assert isinstance(dataset, PiPLSDataset)
    assert dataset.data.shape == (46, 14)
    assert dataset.target.shape == (46, 8)
    assert dataset.data.dtype == np.float64
    assert dataset.target.dtype == np.float64
    assert dataset.feature_names == FEATURE_NAMES
    assert dataset.target_names == TARGET_NAMES
    assert dataset.sample_ids == tuple(f"pulp-{index:02d}" for index in range(1, 47))
    assert not dataset.data.flags.writeable
    assert not dataset.target.flags.writeable
    assert np.isfinite(dataset.data).all()
    assert np.isfinite(dataset.target).all()

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

    np.testing.assert_array_equal(X, dataset.data)
    np.testing.assert_array_equal(Y, dataset.target)
    assert not X.flags.writeable
    assert not Y.flags.writeable
    assert not np.shares_memory(X, second_X)
    assert not np.shares_memory(Y, second_Y)

    with pytest.raises(ValueError):
        X[0, 0] = 0.0
    with pytest.raises(ValueError):
        Y[0, 0] = 0.0


def test_packaged_pulp_resources_match_temporary_repository_copy() -> None:
    dataset = load_pulp()

    np.testing.assert_array_equal(dataset.data, _repository_matrix("X.csv"))
    np.testing.assert_array_equal(dataset.target, _repository_matrix("Y.csv"))

    resource_root = resources.files("pipls").joinpath("_data", "pulp")
    for name, expected_hash in RESOURCE_HASHES.items():
        packaged = resource_root.joinpath(name).read_bytes()
        repository = (_repository_root() / "datasets" / "pulp" / name).read_bytes()
        assert packaged == repository
        assert hashlib.sha256(packaged).hexdigest() == expected_hash

    for name in ("metadata.json", "README.md"):
        assert resource_root.joinpath(name).is_file()


def test_load_pulp_result_is_pickleable() -> None:
    dataset = load_pulp()
    restored = pickle.loads(pickle.dumps(dataset))

    assert isinstance(restored, PiPLSDataset)
    np.testing.assert_array_equal(restored.data, dataset.data)
    np.testing.assert_array_equal(restored.target, dataset.target)
    assert restored.feature_names == dataset.feature_names
    assert restored.target_names == dataset.target_names
    assert restored.sample_ids == dataset.sample_ids
    assert restored.provenance == dataset.provenance
    assert restored.metadata == dataset.metadata
    assert not restored.data.flags.writeable
    assert not restored.target.flags.writeable
