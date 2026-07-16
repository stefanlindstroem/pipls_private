from __future__ import annotations

import numpy as np
import pytest

from pipls import PiPLSPathCV, PiPLSRegression


def _data() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(20260716)
    X = rng.normal(size=(80, 20))
    B = rng.normal(size=(20, 3))
    Y = X @ B + 0.1 * rng.normal(size=(80, 3))
    return X, Y


def _splits() -> list[tuple[np.ndarray, np.ndarray]]:
    return [
        (np.arange(0, 60), np.arange(60, 80)),
        (np.arange(20, 80), np.arange(0, 20)),
    ]


@pytest.mark.parametrize("search_method", ["optimal", "auto"])
def test_fixed_component_path_matches_regression_rank_search(search_method: str) -> None:
    X, Y = _data()
    splits = _splits()
    regression = PiPLSRegression(
        n_components=1,
        predictor_rank=search_method,  # type: ignore[arg-type]
        samples_per_predictor_rank=5,
        cv=splits,
        n_jobs=1,
        svd_solver="full",
        random_state=None,
    ).fit(X, Y)
    path = PiPLSPathCV(
        estimator=PiPLSRegression(
            scale=True,
            svd_solver="full",
            random_state=None,
        ),
        n_components_values=[1],
        predictor_rank_values=list(range(1, 13)),
        max_predictor_rank=12,
        search_method=search_method,  # type: ignore[arg-type]
        cv=splits,
        refit=False,
        n_jobs=1,
    ).fit(X, Y)

    path_ranks = path.cv_results_["predictor_rank"]
    np.testing.assert_array_equal(regression.predictor_rank_values_, path_ranks)
    np.testing.assert_allclose(
        regression.predictor_rank_cv_results_["mean_test_score"],
        path.cv_results_["mean_test_score"],
    )
    np.testing.assert_allclose(
        regression.predictor_rank_cv_results_["mean_response_standardized_mse"],
        path.cv_results_["mean_response_standardized_mse"],
    )
    assert regression.predictor_rank_ == path.best_predictor_rank_
    assert regression.best_score_ == pytest.approx(path.best_score_)


def test_adaptive_fixed_component_search_has_identical_history() -> None:
    X, Y = _data()
    splits = _splits()
    regression = PiPLSRegression(
        n_components=1,
        predictor_rank="auto",
        samples_per_predictor_rank=5,
        cv=splits,
        n_jobs=1,
        svd_solver="full",
        random_state=None,
    ).fit(X, Y)
    path = PiPLSPathCV(
        estimator=PiPLSRegression(svd_solver="full", random_state=None),
        n_components_values=[1],
        predictor_rank_values=list(range(1, 13)),
        max_predictor_rank=12,
        search_method="auto",
        cv=splits,
        refit=False,
        n_jobs=1,
    ).fit(X, Y)

    regression_history = tuple(
        tuple(int(rank) for rank in batch)
        for batch in regression.predictor_rank_search_history_
    )
    assert regression_history == path.path_search_history_[1]
    np.testing.assert_array_equal(
        regression.predictor_rank_evaluation_order_,
        np.concatenate(
            [np.asarray(batch, dtype=np.intp) for batch in path.path_search_history_[1]]
        ),
    )
