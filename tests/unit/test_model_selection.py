from __future__ import annotations

import numpy as np
import pytest
from sklearn.model_selection import KFold

from pipls.model_selection import (
    _materialize_cv_splits,
    _max_predictor_rank,
    _predictor_rank_values,
    _response_standardized_mse,
    _select_predictor_rank,
    _training_response_scale,
)


def test_max_predictor_rank_uses_ceiling_rule() -> None:
    assert (
        _max_predictor_rank(
            n_features=100,
            n_train_min=51,
            samples_per_predictor_rank=10,
        )
        == 6
    )


def test_max_predictor_rank_respects_feature_and_training_caps() -> None:
    assert (
        _max_predictor_rank(
            n_features=3,
            n_train_min=100,
            samples_per_predictor_rank=10,
        )
        == 3
    )
    assert (
        _max_predictor_rank(
            n_features=100,
            n_train_min=4,
            samples_per_predictor_rank=0.1,
        )
        == 4
    )


def test_smaller_training_fold_cannot_increase_rank_bound() -> None:
    complete_data_bound = _max_predictor_rank(
        n_features=30,
        n_train_min=50,
        samples_per_predictor_rank=10,
    )
    fold_bound = _max_predictor_rank(
        n_features=30,
        n_train_min=39,
        samples_per_predictor_rank=10,
    )

    assert complete_data_bound == 5
    assert fold_bound == 4


@pytest.mark.parametrize(
    ("argument", "value"),
    [
        ("n_features", 0),
        ("n_train_min", 0),
        ("samples_per_predictor_rank", 0.0),
        ("samples_per_predictor_rank", np.inf),
        ("samples_per_predictor_rank", True),
    ],
)
def test_max_predictor_rank_rejects_invalid_inputs(argument: str, value: object) -> None:
    kwargs: dict[str, object] = {
        "n_features": 10,
        "n_train_min": 20,
        "samples_per_predictor_rank": 10,
    }
    kwargs[argument] = value

    with pytest.raises(ValueError, match=argument):
        _max_predictor_rank(**kwargs)  # type: ignore[arg-type]


def test_materialize_cv_splits_reuses_one_concrete_split_set() -> None:
    X = np.arange(30, dtype=np.float64).reshape(10, 3)
    y = np.arange(10, dtype=np.float64)

    materialized = _materialize_cv_splits(KFold(n_splits=4), X, y)

    assert len(materialized.splits) == 4
    assert materialized.n_train_min == 7
    np.testing.assert_array_equal(materialized.splits[0][1], np.array([0, 1, 2]))
    assert all(train.dtype == np.intp for train, _ in materialized.splits)
    assert all(validation.dtype == np.intp for _, validation in materialized.splits)


def test_materialize_cv_splits_accepts_an_iterable_and_copies_indices() -> None:
    X = np.zeros((5, 2))
    y = np.zeros((5, 1))
    train = np.array([0, 1, 2], dtype=np.int64)
    validation = np.array([3, 4], dtype=np.int64)

    materialized = _materialize_cv_splits([(train, validation)], X, y)
    train[0] = 4
    validation[0] = 0

    np.testing.assert_array_equal(materialized.splits[0][0], np.array([0, 1, 2]))
    np.testing.assert_array_equal(materialized.splits[0][1], np.array([3, 4]))
    assert materialized.n_train_min == 3


@pytest.mark.parametrize(
    "splits",
    [
        [],
        [([0, 1], [])],
        [([0, 1], [1, 2])],
        [([0, 0], [1, 2])],
        [([0, 1], [2, 5])],
    ],
)
def test_materialize_cv_splits_rejects_invalid_splits(splits: object) -> None:
    X = np.zeros((5, 2))
    y = np.zeros(5)

    with pytest.raises(ValueError):
        _materialize_cv_splits(splits, X, y)


def test_predictor_rank_values_cover_every_admissible_rank() -> None:
    np.testing.assert_array_equal(
        _predictor_rank_values(n_components=3, max_predictor_rank=6),
        np.array([3, 4, 5, 6], dtype=np.intp),
    )


def test_predictor_rank_values_reject_empty_admissible_set() -> None:
    with pytest.raises(ValueError, match="must not exceed"):
        _predictor_rank_values(n_components=4, max_predictor_rank=3)


def test_training_response_scale_uses_sample_standard_deviation() -> None:
    y_train = np.array(
        [
            [1.0, 5.0],
            [3.0, 5.0],
            [5.0, 5.0],
        ]
    )

    scale = _training_response_scale(y_train)

    np.testing.assert_allclose(scale, np.array([2.0, 1.0]))


def test_training_response_scale_is_unit_for_singleton_training_data() -> None:
    np.testing.assert_array_equal(
        _training_response_scale(np.array([[2.0, -3.0]])),
        np.ones(2),
    )


def test_response_standardized_mse_is_uniform_over_samples_and_responses() -> None:
    y_true = np.array([[3.0, 9.0], [5.0, 13.0]])
    y_pred = np.array([[1.0, 5.0], [4.0, 9.0]])
    scale = np.array([2.0, 4.0])

    loss = _response_standardized_mse(y_true, y_pred, scale)

    expected = np.mean(np.array([1.0, 1.0, 0.25, 1.0]))
    assert loss == pytest.approx(expected)


def test_response_standardized_mse_supports_one_dimensional_targets() -> None:
    loss = _response_standardized_mse(
        np.array([2.0, 6.0]),
        np.array([0.0, 4.0]),
        np.array([2.0]),
    )

    assert loss == pytest.approx(1.0)


def test_response_standardized_mse_rejects_invalid_scale() -> None:
    with pytest.raises(ValueError, match="positive finite"):
        _response_standardized_mse(
            np.ones((2, 2)),
            np.zeros((2, 2)),
            np.array([1.0, 0.0]),
        )


def test_select_predictor_rank_uses_low_rank_tie_breaking() -> None:
    rank, loss = _select_predictor_rank(
        np.array([5, 2, 4], dtype=np.intp),
        np.array([0.7, 0.5 + 5e-14, 0.5]),
    )

    assert rank == 2
    assert loss == pytest.approx(0.5 + 5e-14)


def test_select_predictor_rank_prefers_strictly_lower_loss_outside_tolerance() -> None:
    rank, loss = _select_predictor_rank(
        np.array([2, 3], dtype=np.intp),
        np.array([0.5 + 1e-8, 0.5]),
    )

    assert rank == 3
    assert loss == pytest.approx(0.5)


@pytest.mark.parametrize(
    ("ranks", "losses"),
    [
        ([], []),
        ([1, 1], [0.2, 0.3]),
        ([1, 2], [0.2]),
        ([1, 2], [0.2, np.nan]),
    ],
)
def test_select_predictor_rank_rejects_invalid_surfaces(
    ranks: list[int],
    losses: list[float],
) -> None:
    with pytest.raises(ValueError):
        _select_predictor_rank(ranks, losses)
