from __future__ import annotations

import inspect

import pytest

import pipls.datasets as dataset_api
from pipls.datasets import (
    PiPLSDataset,
    SyntheticDataTruth,
    make_pipls_regression,
    make_synthetic_data,
)

_PUBLIC_DATASET_NAMES = [
    "PiPLSDataset",
    "SyntheticDataTruth",
    "PiPLSRegressionTruth",
    "load_pulp",
    "load_sugarcane",
    "load_tobacco",
    "make_synthetic_data",
    "make_pipls_regression",
]


def test_dataset_module_is_a_stable_public_facade() -> None:
    assert dataset_api.__all__ == _PUBLIC_DATASET_NAMES
    for name in _PUBLIC_DATASET_NAMES:
        assert getattr(dataset_api, name).__module__ == "pipls.datasets"


def test_dataset_api_is_exposed_from_pipls_datasets_namespace() -> None:
    assert inspect.signature(make_synthetic_data).parameters["random_state"].default == 0
    assert inspect.signature(make_pipls_regression).parameters["random_state"].default == 0

    dataset = make_synthetic_data(
        n_samples=12,
        n_features=5,
        n_targets=3,
        n_shared=1,
    )
    assert isinstance(dataset, PiPLSDataset)


@pytest.mark.parametrize(
    ("parameter", "value", "error"),
    [
        ("n_samples", 0, ValueError),
        ("n_samples", True, TypeError),
        ("n_features", 0, ValueError),
        ("n_targets", 2.0, TypeError),
        ("n_shared", -1, ValueError),
        ("n_predictor_specific", True, TypeError),
        ("n_response_specific", -1, ValueError),
        ("noise", -0.1, ValueError),
        ("noise", (0.1, -0.1), ValueError),
        ("random_state", None, TypeError),
        ("random_state", 2**32, ValueError),
    ],
)
def test_synthetic_generator_rejects_invalid_public_controls(
    parameter: str,
    value: object,
    error: type[Exception],
) -> None:
    values: dict[str, object] = {
        "n_samples": 12,
        "n_features": 5,
        "n_targets": 4,
        "n_shared": 2,
        "n_predictor_specific": 1,
        "n_response_specific": 1,
    }
    values[parameter] = value
    with pytest.raises(error):
        make_synthetic_data(**values)  # type: ignore[arg-type]


def test_synthetic_generator_rejects_latent_dimensions_larger_than_spaces() -> None:
    with pytest.raises(ValueError, match="n_features"):
        make_synthetic_data(
            n_samples=10,
            n_features=2,
            n_targets=5,
            n_shared=2,
            n_predictor_specific=1,
        )
    with pytest.raises(ValueError, match="n_targets"):
        make_synthetic_data(
            n_samples=10,
            n_features=5,
            n_targets=2,
            n_shared=2,
            n_response_specific=1,
        )


def test_synthetic_generator_is_public_and_returns_its_truth_record() -> None:
    dataset = make_synthetic_data(
        n_samples=1,
        n_features=3,
        n_targets=2,
        n_shared=1,
        random_state=4,
    )

    assert isinstance(dataset, PiPLSDataset)
    assert isinstance(dataset.truth, SyntheticDataTruth)
    assert dataset.metadata["generator"] == "make_synthetic_data"
