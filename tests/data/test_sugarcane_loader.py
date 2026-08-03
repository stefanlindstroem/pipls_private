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
from pipls.datasets import PiPLSDataset, load_sugarcane

FEATURE_NAMES = tuple(str(wavelength) for wavelength in range(780, 2501))
TARGET_NAMES = ("TS", "CP", "ADF", "IVOMD")
RESOURCE_HASHES = {
    "X.csv": "f5437f54fcf2bb0cb4754cfb005836f472bbe5f3b29113e0a78ba4d9973b91c4",
    "Y.csv": "a707650407e86af4fc08448f78fda1ee75ad0de25ee6906e17f159a7f2ec9d93",
    "LICENSE.txt": "4a052472a6f2a041c62eb1530b571f104ea587b662830410e57cfb134ccbe2df",
}
ARRAY_HASHES = {
    "data": "42abcf76dbcb6e9b244dfe8cfdac253dd67273cfeac7244ac79deede3112b0dd",
    "target": "eee177285c6dbdd2b8df8c97a2e3712d1b82b1946f4f9d93229204ebb90d6f8f",
}


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _canonical_array_hash(array: np.ndarray) -> str:
    canonical = np.asarray(array, dtype=np.dtype("<f8"), order="C")
    return hashlib.sha256(canonical.tobytes(order="C")).hexdigest()


def test_load_sugarcane_has_named_loader_return_contract() -> None:
    signature = inspect.signature(load_sugarcane)

    assert tuple(signature.parameters) == ("return_X_y",)
    assert signature.parameters["return_X_y"].kind is inspect.Parameter.KEYWORD_ONLY
    assert signature.parameters["return_X_y"].default is False
    assert "load_sugarcane" in pipls.datasets.__all__
    assert not hasattr(pipls, "load_sugarcane")

    with pytest.raises(TypeError, match="return_X_y must be a boolean"):
        load_sugarcane(return_X_y=1)  # type: ignore[arg-type]


def test_load_sugarcane_returns_labeled_immutable_dataset() -> None:
    dataset = load_sugarcane()

    assert isinstance(dataset, PiPLSDataset)
    assert dataset.data.shape == (57, 1721)
    assert dataset.target.shape == (57, 4)
    assert dataset.data.dtype == np.float64
    assert dataset.target.dtype == np.float64
    assert dataset.feature_names == FEATURE_NAMES
    assert dataset.target_names == TARGET_NAMES
    assert dataset.sample_ids == tuple(
        f"sugarcane-{index:02d}" for index in range(1, 58)
    )
    assert not dataset.data.flags.writeable
    assert not dataset.target.flags.writeable
    assert np.isfinite(dataset.data).all()
    assert np.isfinite(dataset.target).all()

    assert dataset.provenance["source"].endswith("10.17632/mjttsjfj2s.1")
    assert dataset.provenance["license"] == "CC-BY-4.0"
    assert "Chaix" in dataset.provenance["citation"]
    assert dataset.provenance["version"] == "1"

    assert isinstance(dataset.metadata, Mapping)
    assert dataset.metadata["schema_version"] == 1
    assert dataset.metadata["feature_names"] == FEATURE_NAMES
    assert dataset.metadata["target_names"] == TARGET_NAMES
    assert dataset.metadata["dimensions"] == {
        "n_samples": 57,
        "n_features": 1721,
        "n_targets": 4,
    }
    sample_alignment = dataset.metadata["sample_alignment"]
    assert isinstance(sample_alignment, Mapping)
    assert sample_alignment["source_sample_ids"] == tuple(
        sample for sample in range(100, 160) if sample not in {103, 105, 111}
    )
    assert sample_alignment["excluded_source_sample_ids"] == (103, 105, 111)
    with pytest.raises(TypeError):
        dataset.metadata["new"] = "value"  # type: ignore[index]


def test_load_sugarcane_return_X_y_matches_default_result_and_is_fresh() -> None:
    dataset = load_sugarcane()
    second_dataset = load_sugarcane()
    X, Y = load_sugarcane(return_X_y=True)
    second_X, second_Y = load_sugarcane(return_X_y=True)

    assert not np.shares_memory(dataset.data, second_dataset.data)
    assert not np.shares_memory(dataset.target, second_dataset.target)
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


def test_packaged_sugarcane_resources_match_temporary_repository_copy() -> None:
    resource_root = resources.files("pipls").joinpath("_data").joinpath("sugarcane")
    repository_root = _repository_root() / "datasets" / "sugarcane"

    for name, expected_hash in RESOURCE_HASHES.items():
        packaged = resource_root.joinpath(name).read_bytes()
        repository = (repository_root / name).read_bytes()
        assert packaged == repository
        assert hashlib.sha256(packaged).hexdigest() == expected_hash

    for name in ("metadata.json", "README.md"):
        assert resource_root.joinpath(name).is_file()

    dataset = load_sugarcane()
    repository_X = np.loadtxt(repository_root / "X.csv", delimiter=",", skiprows=1)
    repository_Y = np.loadtxt(repository_root / "Y.csv", delimiter=",", skiprows=1)
    np.testing.assert_array_equal(dataset.data, repository_X)
    np.testing.assert_array_equal(dataset.target, repository_Y)
    assert _canonical_array_hash(dataset.data) == ARRAY_HASHES["data"]
    assert _canonical_array_hash(dataset.target) == ARRAY_HASHES["target"]


def test_load_sugarcane_result_is_pickleable() -> None:
    dataset = load_sugarcane()
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
