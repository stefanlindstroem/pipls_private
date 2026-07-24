from __future__ import annotations

import pickle
from collections.abc import Mapping

import numpy as np
import pytest

from pipls.datasets import PiPLSDataset, PiPLSSyntheticTruth, make_pipls_regression

PROVENANCE = {
    "source": "unit-test",
    "license": "BSD-3-Clause",
    "citation": "Synthetic unit-test data",
    "version": "1",
}


def _dataset(**overrides: object) -> PiPLSDataset:
    values: dict[str, object] = {
        "X": np.arange(12, dtype=np.float64).reshape(4, 3),
        "Y": np.arange(8, dtype=np.float64).reshape(4, 2),
        "feature_names": ("a", "b", "c"),
        "target_names": ("u", "v"),
        "sample_ids": ("s0", "s1", "s2", "s3"),
        "provenance": PROVENANCE,
        "metadata": {"nested": {"values": [1, 2]}, "array": np.array([1.0, 2.0])},
    }
    values.update(overrides)
    return PiPLSDataset(**values)  # type: ignore[arg-type]


def test_dataset_normalizes_single_target_and_exposes_sklearn_aliases() -> None:
    dataset = _dataset(
        Y=np.arange(4, dtype=np.float64),
        target_names=("response",),
    )

    assert dataset.X.shape == (4, 3)
    assert dataset.Y.shape == (4, 1)
    assert dataset.data is dataset.X
    assert dataset.target is dataset.Y
    assert dataset.n_samples == 4
    assert dataset.n_features == 3
    assert dataset.n_targets == 1


def test_dataset_copies_and_freezes_arrays_and_metadata() -> None:
    X = np.arange(12, dtype=np.float64).reshape(4, 3)
    metadata_array = np.array([1.0, 2.0])
    dataset = _dataset(X=X, metadata={"array": metadata_array, "items": [1, 2]})

    X[0, 0] = -999.0
    metadata_array[0] = -999.0

    assert dataset.X[0, 0] == 0.0
    assert not dataset.X.flags.writeable
    assert not dataset.Y.flags.writeable
    assert isinstance(dataset.metadata, Mapping)
    frozen_array = dataset.metadata["array"]
    assert isinstance(frozen_array, np.ndarray)
    assert frozen_array[0] == 1.0
    assert not frozen_array.flags.writeable
    assert dataset.metadata["items"] == (1, 2)
    with pytest.raises(TypeError):
        dataset.metadata["new"] = 1  # type: ignore[index]

    restored = pickle.loads(pickle.dumps(dataset))
    assert isinstance(restored, PiPLSDataset)
    np.testing.assert_array_equal(restored.X, dataset.X)
    assert restored.metadata["items"] == (1, 2)


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("X", [["a"]], TypeError),
        ("X", np.ones(4), ValueError),
        ("X", np.array([[np.nan], [1.0], [2.0], [3.0]]), ValueError),
        ("Y", np.ones((3, 2)), ValueError),
        ("feature_names", ("a", "b"), ValueError),
        ("feature_names", ("a", "a", "c"), ValueError),
        ("target_names", ("u", ""), TypeError),
        ("sample_ids", ("s0", "s1", "s1", "s3"), ValueError),
        ("provenance", {"source": "x"}, ValueError),
        ("metadata", {"bad": object()}, TypeError),
    ],
)
def test_dataset_rejects_invalid_public_inputs(
    field: str,
    value: object,
    error: type[Exception],
) -> None:
    with pytest.raises(error):
        _dataset(**{field: value})


def test_dataset_accepts_and_freezes_non_object_metadata_arrays() -> None:
    source_arrays = {
        "booleans": np.array([True, False]),
        "strings": np.array(["alpha", "beta"]),
        "dates": np.array(["2026-01-01", "2026-01-02"], dtype="datetime64[D]"),
    }
    dataset = _dataset(metadata=source_arrays)

    source_arrays["booleans"][0] = False
    source_arrays["strings"][0] = "changed"
    source_arrays["dates"][0] = np.datetime64("2030-01-01")

    for name, expected in (
        ("booleans", np.array([True, False])),
        ("strings", np.array(["alpha", "beta"])),
        ("dates", np.array(["2026-01-01", "2026-01-02"], dtype="datetime64[D]")),
    ):
        frozen = dataset.metadata[name]
        assert isinstance(frozen, np.ndarray)
        assert not frozen.flags.writeable
        np.testing.assert_array_equal(frozen, expected)


@pytest.mark.parametrize(
    "metadata_array",
    [
        np.array([{"items": []}], dtype=object),
        np.array([[1, 2], [3, 4]], dtype=object),
        np.array([bytearray(b"mutable")], dtype=object),
    ],
)
def test_dataset_rejects_object_dtype_metadata_arrays(
    metadata_array: np.ndarray,
) -> None:
    with pytest.raises(TypeError, match="object-dtype NumPy array"):
        _dataset(metadata={"nested": {"array": metadata_array}})


def test_synthetic_truth_is_read_only() -> None:
    dataset = make_pipls_regression(
        n_samples=12,
        n_features=6,
        n_targets=4,
        n_shared=2,
        n_predictor_specific=1,
        n_response_specific=1,
        random_state=7,
    )
    truth = dataset.truth

    assert isinstance(truth, PiPLSSyntheticTruth)
    assert truth.n_shared == 2
    assert truth.n_predictor_specific == 1
    assert truth.n_response_specific == 1
    for name in truth.__dataclass_fields__:
        assert not getattr(truth, name).flags.writeable
