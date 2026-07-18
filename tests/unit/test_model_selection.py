from __future__ import annotations

import numpy as np
import pytest
from sklearn.model_selection import KFold

from pipls.model_selection import (
    _adaptive_refinement_interval,
    _logarithmic_predictor_rank_values,
    _materialize_cv_splits,
    _max_predictor_rank,
    _select_predictor_rank,
)


def test_max_predictor_rank_uses_ceiling_rule() -> None:
    assert (
        _max_predictor_rank(
            n_features=100,
            n_samples=51,
            n_train_min=51,
            samples_per_predictor_rank=10,
        )
        == 6
    )


def test_max_predictor_rank_respects_feature_and_training_caps() -> None:
    assert (
        _max_predictor_rank(
            n_features=3,
            n_samples=100,
            n_train_min=100,
            samples_per_predictor_rank=10,
        )
        == 3
    )
    assert (
        _max_predictor_rank(
            n_features=100,
            n_samples=100,
            n_train_min=4,
            samples_per_predictor_rank=0.1,
        )
        == 3
    )


def test_support_term_uses_total_samples_while_fold_size_caps_feasibility() -> None:
    complete_data_bound = _max_predictor_rank(
        n_features=30,
        n_samples=50,
        n_train_min=50,
        samples_per_predictor_rank=10,
    )
    fold_bound = _max_predictor_rank(
        n_features=30,
        n_samples=50,
        n_train_min=39,
        samples_per_predictor_rank=10,
    )
    feasibility_capped = _max_predictor_rank(
        n_features=30,
        n_samples=50,
        n_train_min=4,
        samples_per_predictor_rank=10,
    )

    assert complete_data_bound == 5
    assert fold_bound == 5
    assert feasibility_capped == 3


@pytest.mark.parametrize(
    ("argument", "value"),
    [
        ("n_features", 0),
        ("n_samples", 0),
        ("n_train_min", 0),
        ("samples_per_predictor_rank", 0.0),
        ("samples_per_predictor_rank", np.inf),
        ("samples_per_predictor_rank", True),
    ],
)
def test_max_predictor_rank_rejects_invalid_inputs(argument: str, value: object) -> None:
    kwargs: dict[str, object] = {
        "n_features": 10,
        "n_samples": 20,
        "n_train_min": 20,
        "samples_per_predictor_rank": 10,
    }
    kwargs[argument] = value

    with pytest.raises(ValueError, match=argument):
        _max_predictor_rank(**kwargs)  # type: ignore[arg-type]


def test_max_predictor_rank_rejects_training_fold_larger_than_full_data() -> None:
    with pytest.raises(ValueError, match="n_train_min must not exceed n_samples"):
        _max_predictor_rank(
            n_features=10,
            n_samples=19,
            n_train_min=20,
            samples_per_predictor_rank=5,
        )


def test_max_predictor_rank_rejects_singleton_training_folds() -> None:
    with pytest.raises(ValueError, match="at least 2"):
        _max_predictor_rank(
            n_features=10,
            n_samples=20,
            n_train_min=1,
            samples_per_predictor_rank=5,
        )


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


def test_logarithmic_predictor_rank_values_are_deterministic_and_include_endpoints() -> None:
    values = _logarithmic_predictor_rank_values(lower=2, upper=100)

    np.testing.assert_array_equal(values, np.array([2, 4, 7, 14, 27, 52, 100]))
    assert values[0] == 2
    assert values[-1] == 100
    assert np.all(np.diff(values) > 0)


def test_logarithmic_predictor_rank_values_handle_small_intervals() -> None:
    np.testing.assert_array_equal(
        _logarithmic_predictor_rank_values(lower=4, upper=4),
        np.array([4]),
    )


def test_adaptive_refinement_interval_uses_neighbors_around_best_rank() -> None:
    interval = _adaptive_refinement_interval(
        np.array([2, 4, 8, 16, 32]),
        np.array([5.0, 3.0, 1.0, 2.0, 4.0]),
    )

    assert interval == (4, 16)


def test_adaptive_refinement_interval_respects_lower_rank_ties() -> None:
    interval = _adaptive_refinement_interval(
        np.array([2, 4, 8, 16]),
        np.array([1.0, 1.0, 2.0, 3.0]),
    )

    assert interval == (2, 4)


def test_rank_test_scores_assigns_minimum_rank_to_ties() -> None:
    from pipls.model_selection import _rank_test_scores

    ranks = _rank_test_scores(np.array([0.5, 0.5, 0.2, 0.1, 0.2]))
    np.testing.assert_array_equal(ranks, np.array([1, 1, 3, 5, 3]))
