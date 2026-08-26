from __future__ import annotations

import numpy as np
from numpy.testing import assert_array_equal

from pipls.datasets import (
    SyntheticDataTruth,
    make_pipls_regression,
    make_pipls_train_test,
    make_synthetic_data,
)

PARAMETERS = {
    "n_samples": 24,
    "n_features": 7,
    "n_targets": 4,
    "n_shared": 2,
    "n_predictor_specific": 2,
    "n_response_specific": 1,
    "shared_distribution": "uniform",
    "predictor_specific_distribution": "normal",
    "response_specific_distribution": "uniform",
    "noise": (0.05, 0.1),
}


def test_synthetic_generator_is_exactly_reproducible() -> None:
    first = make_pipls_regression(**PARAMETERS, random_state=314)
    second = make_pipls_regression(**PARAMETERS, random_state=np.int64(314))

    assert_array_equal(first.X, second.X)
    assert_array_equal(first.Y, second.Y)
    assert first.feature_names == second.feature_names
    assert first.target_names == second.target_names
    assert first.sample_ids == second.sample_ids


def test_different_seed_changes_generated_values() -> None:
    first = make_pipls_regression(**PARAMETERS, random_state=314)
    second = make_pipls_regression(**PARAMETERS, random_state=315)

    assert not np.array_equal(first.X, second.X)
    assert not np.array_equal(first.Y, second.Y)


def test_train_block_is_independent_of_requested_test_size() -> None:
    common = {
        "n_train": 24,
        "n_features": 7,
        "n_targets": 4,
        "n_shared": 2,
        "n_predictor_specific": 2,
        "n_response_specific": 1,
        "random_state": 88,
    }
    train_small, _ = make_pipls_train_test(**common, n_test=5)
    train_large, _ = make_pipls_train_test(**common, n_test=50)

    assert_array_equal(train_small.X, train_large.X)
    assert_array_equal(train_small.Y, train_large.Y)



def test_manuscript_generator_is_exactly_reproducible() -> None:
    parameters = {
        "n_samples": 17,
        "n_features": 6,
        "n_targets": 5,
        "n_shared": 2,
        "n_predictor_specific": 1,
        "n_response_specific": 2,
        "noise": (0.15, 0.25),
    }
    first = make_synthetic_data(**parameters, random_state=271)
    second = make_synthetic_data(**parameters, random_state=np.int64(271))

    assert_array_equal(first.X, second.X)
    assert_array_equal(first.Y, second.Y)
    first_truth = first.truth
    second_truth = second.truth
    assert isinstance(first_truth, SyntheticDataTruth)
    assert isinstance(second_truth, SyntheticDataTruth)
    for name in first_truth.__dataclass_fields__:
        assert_array_equal(getattr(first_truth, name), getattr(second_truth, name))


def test_manuscript_generator_matches_documented_rng_draw_order() -> None:
    n_samples = 7
    n_features = 5
    n_targets = 4
    n_predictor_specific = 2
    n_shared = 1
    n_response_specific = 2
    sigma_x = 0.3
    sigma_y = 0.4
    seed = 93
    rng = np.random.default_rng(seed)

    expected = {
        "predictor_specific_scores": rng.standard_normal(
            (n_samples, n_predictor_specific)
        ),
        "shared_scores": rng.standard_normal((n_samples, n_shared)),
        "response_specific_scores": rng.standard_normal(
            (n_samples, n_response_specific)
        ),
        "predictor_specific_loadings": rng.standard_normal(
            (n_predictor_specific, n_features)
        ),
        "shared_predictor_loadings": rng.standard_normal((n_shared, n_features)),
        "shared_response_loadings": rng.standard_normal((n_shared, n_targets)),
        "response_specific_loadings": rng.standard_normal(
            (n_response_specific, n_targets)
        ),
        "x_noise": sigma_x * rng.standard_normal((n_samples, n_features)),
        "y_noise": sigma_y * rng.standard_normal((n_samples, n_targets)),
    }
    dataset = make_synthetic_data(
        n_samples=n_samples,
        n_features=n_features,
        n_targets=n_targets,
        n_shared=n_shared,
        n_predictor_specific=n_predictor_specific,
        n_response_specific=n_response_specific,
        noise=(sigma_x, sigma_y),
        random_state=seed,
    )
    truth = dataset.truth
    assert isinstance(truth, SyntheticDataTruth)

    for name, values in expected.items():
        assert_array_equal(getattr(truth, name), values)
    expected_x_signal = (
        expected["predictor_specific_scores"]
        @ expected["predictor_specific_loadings"]
        + expected["shared_scores"] @ expected["shared_predictor_loadings"]
    )
    expected_y_signal = (
        expected["shared_scores"] @ expected["shared_response_loadings"]
        + expected["response_specific_scores"]
        @ expected["response_specific_loadings"]
    )
    assert_array_equal(truth.x_signal, expected_x_signal)
    assert_array_equal(truth.y_signal, expected_y_signal)
    assert_array_equal(dataset.X, expected_x_signal + expected["x_noise"])
    assert_array_equal(dataset.Y, expected_y_signal + expected["y_noise"])
