from __future__ import annotations

import numpy as np
from numpy.testing import assert_array_equal

from pipls.datasets import make_pipls_regression, make_pipls_train_test

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
