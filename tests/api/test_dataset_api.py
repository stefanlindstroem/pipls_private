from __future__ import annotations

import inspect

import numpy as np
import pytest

from pipls.datasets import (
    PiPLSDataset,
    PiPLSLatentGeometryTruth,
    make_pipls_latent_geometry,
    make_pipls_regression,
    make_pipls_train_test,
)


def test_dataset_api_is_exposed_from_pipls_datasets_namespace() -> None:
    assert inspect.signature(make_pipls_latent_geometry).parameters["random_state"].default == 0
    assert inspect.signature(make_pipls_regression).parameters["random_state"].default == 0
    assert inspect.signature(make_pipls_train_test).parameters["random_state"].default == 0

    dataset = make_pipls_regression(
        n_samples=12,
        n_features=5,
        n_targets=3,
        n_shared=1,
    )
    assert isinstance(dataset, PiPLSDataset)


@pytest.mark.parametrize(
    ("parameter", "value", "error"),
    [
        ("n_samples", True, TypeError),
        ("n_samples", 1, ValueError),
        ("n_features", 0, ValueError),
        ("n_targets", 1.0, TypeError),
        ("n_shared", -1, ValueError),
        ("n_predictor_specific", True, TypeError),
        ("n_response_specific", -1, ValueError),
        ("shared_distribution", "gamma", ValueError),
        ("shared_strength", 0.0, ValueError),
        ("feature_scale", [1.0, 2.0], ValueError),
        ("target_scale", np.inf, ValueError),
        ("noise", -0.1, ValueError),
        ("noise", (0.1, -0.1), ValueError),
        ("random_state", None, TypeError),
        ("random_state", 2**32, ValueError),
    ],
)
def test_generator_rejects_invalid_public_controls(
    parameter: str,
    value: object,
    error: type[Exception],
) -> None:
    values: dict[str, object] = {
        "n_samples": 12,
        "n_features": 5,
        "n_targets": 3,
        "n_shared": 1,
        "n_predictor_specific": 1,
        "n_response_specific": 1,
    }
    values[parameter] = value
    with pytest.raises(error):
        make_pipls_regression(**values)  # type: ignore[arg-type]


def test_generator_rejects_sample_blocks_too_small_for_centered_latent_rank() -> None:
    with pytest.raises(ValueError, match="centered latent rank"):
        make_pipls_regression(
            n_samples=3,
            n_features=5,
            n_targets=4,
            n_shared=2,
            n_predictor_specific=1,
        )
    with pytest.raises(ValueError, match="n_test"):
        make_pipls_train_test(
            n_train=10,
            n_test=3,
            n_features=5,
            n_targets=4,
            n_shared=2,
            n_predictor_specific=1,
        )


def test_generator_rejects_latent_dimensions_larger_than_observed_spaces() -> None:
    with pytest.raises(ValueError, match="n_features"):
        make_pipls_regression(
            n_samples=10,
            n_features=2,
            n_targets=5,
            n_shared=2,
            n_predictor_specific=1,
        )
    with pytest.raises(ValueError, match="n_targets"):
        make_pipls_regression(
            n_samples=10,
            n_features=5,
            n_targets=2,
            n_shared=2,
            n_response_specific=1,
        )



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
def test_manuscript_generator_rejects_invalid_public_controls(
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
        make_pipls_latent_geometry(**values)  # type: ignore[arg-type]


def test_manuscript_generator_rejects_latent_dimensions_larger_than_spaces() -> None:
    with pytest.raises(ValueError, match="n_features"):
        make_pipls_latent_geometry(
            n_samples=10,
            n_features=2,
            n_targets=5,
            n_shared=2,
            n_predictor_specific=1,
        )
    with pytest.raises(ValueError, match="n_targets"):
        make_pipls_latent_geometry(
            n_samples=10,
            n_features=5,
            n_targets=2,
            n_shared=2,
            n_response_specific=1,
        )


def test_manuscript_generator_is_public_and_returns_its_truth_record() -> None:
    dataset = make_pipls_latent_geometry(
        n_samples=1,
        n_features=3,
        n_targets=2,
        n_shared=1,
        random_state=4,
    )

    assert isinstance(dataset, PiPLSDataset)
    assert isinstance(dataset.truth, PiPLSLatentGeometryTruth)
    assert dataset.metadata["generator"] == "make_pipls_latent_geometry"
