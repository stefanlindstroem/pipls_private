from __future__ import annotations

import numpy as np
import pytest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from pipls import PiPLSPathCV, PiPLSRegression, StatisticalSupportWarning


def _data(n_samples: int = 36) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(20260716)
    X = rng.normal(size=(n_samples, 8))
    B = rng.normal(size=(8, 3))
    Y = X @ B + 0.05 * rng.normal(size=(n_samples, 3))
    return X, Y


def test_optimal_path_evaluates_complete_triangular_grid() -> None:
    X, Y = _data()
    search = PiPLSPathCV(
        samples_per_predictor_rank=8,
        cv=3,
        n_jobs=1,
    ).fit(X, Y)

    assert search.max_predictor_rank_ == 5
    assert search.n_path_candidates_ == 12
    assert search.n_path_candidates_evaluated_ == 12
    assert search.n_path_candidates_skipped_ == 0
    assert search.path_search_exhaustive_
    assert search.path_search_method_ == "auto"
    np.testing.assert_array_equal(search.n_components_values_, np.array([1, 2, 3]))
    np.testing.assert_array_equal(
        search.predictor_rank_values_, np.array([1, 2, 3, 4, 5])
    )
    assert np.isnan(search.response_standardized_mse_path_[1, 0])
    assert np.isnan(search.response_standardized_mse_path_[2, 0])
    assert np.isnan(search.response_standardized_mse_path_[2, 1])


def test_best_estimator_is_refitted_and_delegates_prediction() -> None:
    X, Y = _data()
    search = PiPLSPathCV(
        n_components_values=[1, 2],
        predictor_rank_values=[1, 2, 3],
        cv=3,
        n_jobs=1,
    ).fit(X, Y)

    assert isinstance(search.best_estimator_, PiPLSRegression)
    assert search.best_estimator_.n_components == search.best_n_components_
    assert search.best_estimator_.predictor_rank == search.best_predictor_rank_
    np.testing.assert_allclose(search.predict(X), search.best_estimator_.predict(X))
    assert search.score(X, Y) == pytest.approx(search.best_estimator_.score(X, Y))


def test_refit_false_disables_prediction() -> None:
    X, Y = _data()
    search = PiPLSPathCV(
        n_components_values=[1],
        predictor_rank_values=[1, 2],
        refit=False,
        cv=3,
    ).fit(X, Y)

    assert not hasattr(search, "best_estimator_")
    with pytest.raises(AttributeError, match="refit=False"):
        search.predict(X)


def test_pipeline_is_cloned_inside_each_fold_and_prefix_is_inferred() -> None:
    X, Y = _data(18)
    splits = [
        (np.arange(0, 12), np.arange(12, 18)),
        (np.arange(6, 18), np.arange(0, 6)),
    ]
    pipeline = Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "regression",
                PiPLSRegression(
                    n_components=1,
                    predictor_rank=1,
                    scale=False,
                    svd_solver="full",
                    random_state=None,
                ),
            ),
        ]
    )
    search = PiPLSPathCV(
        estimator=pipeline,
        n_components_values=[1],
        predictor_rank_values=[1],
        cv=splits,
        n_jobs=1,
    ).fit(X, Y)

    manual = pipeline.fit(X[:12], Y[:12])
    prediction = manual.predict(X[12:18])
    response_scale = np.std(Y[:12], axis=0, ddof=1)
    expected = np.mean(((Y[12:18] - prediction) / response_scale[None, :]) ** 2)

    assert search.pipls_param_prefix_ == "regression"
    assert search.cv_results_["split0_response_standardized_mse"][0] == pytest.approx(expected)
    assert isinstance(search.best_estimator_, Pipeline)


def test_explicit_nested_prefix_is_validated() -> None:
    X, Y = _data()
    pipeline = Pipeline([("regression", PiPLSRegression(predictor_rank=1))])

    with pytest.raises(ValueError, match="does not locate"):
        PiPLSPathCV(
            estimator=pipeline,
            pipls_param_prefix="missing",
            n_components_values=[1],
            predictor_rank_values=[1],
        ).fit(X, Y)


def test_auto_path_skips_candidates_with_constant_scorer() -> None:
    rng = np.random.default_rng(20260717)
    X = rng.normal(size=(80, 20))
    Y = X @ rng.normal(size=(20, 3)) + 0.05 * rng.normal(size=(80, 3))

    def constant_scorer(estimator: object, X_validation: object, y_validation: object) -> float:
        del estimator, X_validation, y_validation
        return 1.0

    search = PiPLSPathCV(
        n_components_values=[1],
        predictor_rank_values=list(range(1, 13)),
        max_predictor_rank=12,
        search_method="auto",
        samples_per_predictor_rank=5,
        cv=4,
        scoring=constant_scorer,
        n_jobs=1,
    ).fit(X, Y)

    assert search.best_n_components_ == 1
    assert search.best_predictor_rank_ == 1
    assert search.n_path_candidates_ == 12
    assert search.n_path_candidates_evaluated_ < 12
    assert search.n_path_candidates_skipped_ > 0
    assert not search.path_search_exhaustive_
    assert search.path_search_history_[1]


def test_global_tie_breaking_prefers_lower_components_then_rank() -> None:
    X, Y = _data()

    def constant_scorer(estimator: object, X_validation: object, y_validation: object) -> float:
        del estimator, X_validation, y_validation
        return 1.0

    search = PiPLSPathCV(
        n_components_values=[1, 2],
        predictor_rank_values=[1, 2, 3],
        scoring=constant_scorer,
        cv=3,
    ).fit(X, Y)

    assert search.best_n_components_ == 1
    assert search.best_predictor_rank_ == 1
    assert search.best_predictor_rank_by_n_components_ == {1: 1, 2: 2}


def test_explicit_max_predictor_rank_bypasses_rule_bound() -> None:
    X, Y = _data()
    search = PiPLSPathCV(
        n_components_values=[1],
        predictor_rank_values=[1, 2, 3],
        max_predictor_rank=3,
        samples_per_predictor_rank=100,
        cv=3,
    ).fit(X, Y)

    assert search.max_predictor_rank_ == 3
    np.testing.assert_array_equal(search.predictor_rank_values_, np.array([1, 2, 3]))


def test_low_samples_per_predictor_rank_warns() -> None:
    X, Y = _data()
    with pytest.warns(StatisticalSupportWarning, match="statistical support"):
        PiPLSPathCV(
            n_components_values=[1],
            predictor_rank_values=[1],
            samples_per_predictor_rank=4,
            cv=3,
        ).fit(X, Y)


@pytest.mark.parametrize(
    ("keyword", "value", "message"),
    [
        ("search_method", "exhaustive", "search_method"),
        ("max_predictor_rank", 0, "max_predictor_rank"),
        ("n_components_values", [], "must not be empty"),
        ("predictor_rank_values", [1.0], "positive integer"),
        ("n_jobs", 0, "must not be zero"),
        ("refit", 1, "refit must be boolean"),
    ],
)
def test_invalid_public_controls_are_rejected(
    keyword: str,
    value: object,
    message: str,
) -> None:
    X, Y = _data()
    kwargs = {keyword: value}
    with pytest.raises(ValueError, match=message):
        PiPLSPathCV(**kwargs).fit(X, Y)
