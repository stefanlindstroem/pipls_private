from __future__ import annotations

import numpy as np
from numpy.testing import assert_allclose

from pipls.datasets import (
    SyntheticDataTruth,
    make_pipls_regression,
    make_pipls_train_test,
    make_synthetic_data,
)


def test_synthetic_shapes_and_reconstruction_identity() -> None:
    dataset = make_pipls_regression(
        n_samples=40,
        n_features=8,
        n_targets=5,
        n_shared=2,
        n_predictor_specific=3,
        n_response_specific=2,
        noise=(0.2, 0.3),
        random_state=42,
    )
    truth = dataset.truth
    assert truth is not None

    assert dataset.X.shape == (40, 8)
    assert dataset.Y.shape == (40, 5)
    assert truth.shared_scores.shape == (40, 2)
    assert truth.predictor_specific_scores.shape == (40, 3)
    assert truth.response_specific_scores.shape == (40, 2)
    assert not hasattr(truth, "x_response_specific_loadings")
    assert not hasattr(truth, "y_predictor_specific_loadings")
    assert_allclose(dataset.X, truth.x_signal + truth.x_noise)
    assert_allclose(dataset.Y, truth.y_signal + truth.y_noise)


def test_noise_free_signal_has_declared_predictor_and_response_ranks() -> None:
    dataset = make_pipls_regression(
        n_samples=80,
        n_features=9,
        n_targets=6,
        n_shared=2,
        n_predictor_specific=3,
        n_response_specific=2,
        shared_strength=(3.0, 1.5),
        predictor_specific_strength=(2.5, 1.2, 0.7),
        response_specific_strength=(1.8, 0.9),
        noise=0.0,
        random_state=123,
    )
    truth = dataset.truth
    assert truth is not None

    assert np.linalg.matrix_rank(dataset.X) == 5
    assert np.linalg.matrix_rank(dataset.Y) == 4
    assert_allclose(dataset.X, truth.x_signal)
    assert_allclose(dataset.Y, truth.y_signal)


def test_train_test_share_model_but_not_sample_realizations() -> None:
    train, test = make_pipls_train_test(
        n_train=30,
        n_test=20,
        n_features=7,
        n_targets=4,
        n_shared=2,
        n_predictor_specific=2,
        n_response_specific=1,
        feature_scale=(1.0, 2.0, 1.0, 0.5, 1.5, 1.0, 0.8),
        target_scale=(1.0, 2.0, 0.5, 1.5),
        random_state=9,
    )
    train_truth = train.truth
    test_truth = test.truth
    assert train_truth is not None and test_truth is not None

    assert train.metadata["split_role"] == "train"
    assert test.metadata["split_role"] == "test"
    assert_allclose(train_truth.x_shared_loadings, test_truth.x_shared_loadings)
    assert_allclose(
        train_truth.x_predictor_specific_loadings,
        test_truth.x_predictor_specific_loadings,
    )
    assert_allclose(train_truth.y_shared_loadings, test_truth.y_shared_loadings)
    assert_allclose(
        train_truth.y_response_specific_loadings,
        test_truth.y_response_specific_loadings,
    )
    assert not np.array_equal(train_truth.shared_scores[:20], test_truth.shared_scores)


def test_generator_does_not_mutate_numpy_global_rng() -> None:
    np.random.seed(124)
    expected = np.random.random(5)
    np.random.seed(124)

    make_pipls_regression(
        n_samples=10,
        n_features=4,
        n_targets=3,
        n_shared=1,
        random_state=5,
    )
    observed = np.random.random(5)

    assert_allclose(observed, expected)



def test_manuscript_latent_geometry_matches_both_generating_equations() -> None:
    dataset = make_synthetic_data(
        n_samples=40,
        n_features=8,
        n_targets=5,
        n_shared=2,
        n_predictor_specific=3,
        n_response_specific=2,
        noise=(0.2, 0.3),
        random_state=42,
    )
    truth = dataset.truth
    assert isinstance(truth, SyntheticDataTruth)

    assert truth.predictor_specific_scores.shape == (40, 3)
    assert truth.shared_scores.shape == (40, 2)
    assert truth.response_specific_scores.shape == (40, 2)
    assert truth.predictor_specific_loadings.shape == (3, 8)
    assert truth.shared_predictor_loadings.shape == (2, 8)
    assert truth.shared_response_loadings.shape == (2, 5)
    assert truth.response_specific_loadings.shape == (2, 5)
    assert_allclose(
        truth.x_signal,
        truth.predictor_specific_scores @ truth.predictor_specific_loadings
        + truth.shared_scores @ truth.shared_predictor_loadings,
    )
    assert_allclose(
        truth.y_signal,
        truth.shared_scores @ truth.shared_response_loadings
        + truth.response_specific_scores @ truth.response_specific_loadings,
    )
    assert_allclose(dataset.X, truth.x_signal + truth.x_noise)
    assert_allclose(dataset.Y, truth.y_signal + truth.y_noise)


def test_manuscript_latent_geometry_supports_structurally_absent_blocks() -> None:
    dataset = make_synthetic_data(
        n_samples=6,
        n_features=4,
        n_targets=3,
        n_shared=0,
        n_predictor_specific=0,
        n_response_specific=0,
        noise=(0.1, 0.2),
        random_state=11,
    )
    truth = dataset.truth
    assert isinstance(truth, SyntheticDataTruth)

    assert truth.predictor_specific_scores.shape == (6, 0)
    assert truth.shared_scores.shape == (6, 0)
    assert truth.response_specific_scores.shape == (6, 0)
    assert truth.predictor_specific_loadings.shape == (0, 4)
    assert truth.shared_predictor_loadings.shape == (0, 4)
    assert truth.shared_response_loadings.shape == (0, 3)
    assert truth.response_specific_loadings.shape == (0, 3)
    assert_allclose(truth.x_signal, 0.0)
    assert_allclose(truth.y_signal, 0.0)


def test_manuscript_generator_does_not_mutate_numpy_global_rng() -> None:
    np.random.seed(318)
    expected = np.random.random(5)
    np.random.seed(318)

    make_synthetic_data(
        n_samples=10,
        n_features=4,
        n_targets=3,
        n_shared=1,
        random_state=8,
    )
    observed = np.random.random(5)

    assert_allclose(observed, expected)
