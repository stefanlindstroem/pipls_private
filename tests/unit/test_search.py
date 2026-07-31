from __future__ import annotations

import pickle
import warnings

import numpy as np
import pytest
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.exceptions import NotFittedError
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted

from pipls import (
    PiPLSComponentPath,
    PiPLSPredictorRankProfile,
    PiPLSRegression,
    PiPLSSearchCV,
    StatisticalSupportWarning,
)
from pipls.metrics import neg_response_standardized_mse


class _WarningTransformer(TransformerMixin, BaseEstimator):  # type: ignore[misc]
    """Small step proving that path fits suppress only package support warnings."""

    def fit(self, X: object, y: object = None) -> _WarningTransformer:
        del X, y
        warnings.warn("unrelated path warning", RuntimeWarning, stacklevel=2)
        return self

    def transform(self, X: object) -> object:
        return X


class _RankTwoTransformer(TransformerMixin, BaseEstimator):  # type: ignore[misc]
    """Return three columns whose numerical rank is at most two."""

    def fit(self, X: object, y: object = None) -> _RankTwoTransformer:
        del X, y
        return self

    def transform(self, X: object) -> np.ndarray:
        array = np.asarray(X, dtype=np.float64)
        return np.column_stack([array[:, 0], array[:, 1], array[:, 0] + array[:, 1]])


def _data(n_samples: int = 36) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(20260716)
    X = rng.normal(size=(n_samples, 8))
    B = rng.normal(size=(8, 3))
    Y = X @ B + 0.05 * rng.normal(size=(n_samples, 3))
    return X, Y


def _rank_two_data(n_samples: int = 30) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(20260723)
    latent = rng.normal(size=(n_samples, 2))
    X = np.column_stack(
        [
            latent[:, 0],
            latent[:, 1],
            latent[:, 0] + latent[:, 1],
            2.0 * latent[:, 0],
            np.ones(n_samples),
        ]
    )
    Y = np.column_stack(
        [
            latent[:, 0] + 0.05 * rng.normal(size=n_samples),
            latent[:, 1] + 0.05 * rng.normal(size=n_samples),
            latent.sum(axis=1) + 0.05 * rng.normal(size=n_samples),
        ]
    )
    return X, Y


def _one_standard_error_data() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    X = rng.normal(size=(24, 6))
    coefficients = np.zeros((6, 3))
    coefficients[:2, :] = rng.normal(size=(2, 3))
    Y = X @ coefficients + rng.normal(size=(24, 3))
    return X, Y


def test_optimal_path_evaluates_complete_triangular_grid() -> None:
    X, Y = _data()
    search = PiPLSSearchCV(
        samples_per_predictor_rank=8,
        cv=3,
        n_jobs=1,
    ).fit(X, Y)

    assert search.max_predictor_rank_ == 5
    assert search.path_search_exhaustive_
    assert search.cv_results_["n_components"].size == 12
    np.testing.assert_array_equal(
        search.component_path_.n_components,
        np.array([1, 2, 3]),
    )
    np.testing.assert_array_equal(
        search.predictor_rank_profile(1).predictor_rank,
        np.array([1, 2, 3, 4, 5]),
    )
    evaluated_pairs = set(
        zip(
            search.cv_results_["n_components"],
            search.cv_results_["predictor_rank"],
            strict=True,
        )
    )
    assert (2, 1) not in evaluated_pairs
    assert (3, 1) not in evaluated_pairs
    assert (3, 2) not in evaluated_pairs


@pytest.mark.parametrize("svd_solver", ["full", "randomized"])
def test_path_caps_candidates_at_minimum_fold_numerical_rank(
    svd_solver: str,
) -> None:
    X, Y = _rank_two_data()
    search = PiPLSSearchCV(
        estimator=PiPLSRegression(
            n_components=1,
            predictor_rank=1,
            svd_solver=svd_solver,
            random_state=0,
        ),
        search_method="optimal",
        cv=3,
        refit=False,
        n_jobs=1,
    ).fit(X, Y)

    assert search.max_predictor_rank_ == 2
    np.testing.assert_array_equal(
        search.component_path_.n_components,
        np.array([1, 2]),
    )
    assert np.max(search.cv_results_["predictor_rank"]) == 2
    assert set(
        zip(
            search.cv_results_["n_components"],
            search.cv_results_["predictor_rank"],
            strict=True,
        )
    ) == {(1, 1), (1, 2), (2, 2)}


def test_path_uses_the_minimum_numerical_rank_across_training_folds() -> None:
    X = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 2.0],
            [2.0, 2.0, 4.0],
            [3.0, 3.0, 6.0],
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
            [1.0, 1.0, 2.0],
        ]
    )
    Y = np.column_stack([X[:, 0] + X[:, 1], X[:, 0] - X[:, 1]])
    splits = (
        (np.arange(0, 4), np.arange(4, 8)),
        (np.arange(4, 8), np.arange(0, 4)),
    )

    search = PiPLSSearchCV(
        estimator=PiPLSRegression(
            n_components=1,
            predictor_rank=1,
            scale=False,
            svd_solver="full",
        ),
        max_predictor_rank=3,
        search_method="optimal",
        cv=splits,
        refit=False,
        n_jobs=1,
    ).fit(X, Y)

    assert search.max_predictor_rank_ == 1
    np.testing.assert_array_equal(search.cv_results_["predictor_rank"], np.array([1]))


def test_pipeline_rank_preflight_uses_fold_local_transformed_predictors() -> None:
    X, Y = _data(30)
    pipeline = Pipeline(
        [
            ("rank_two", _RankTwoTransformer()),
            (
                "regression",
                PiPLSRegression(
                    n_components=1,
                    predictor_rank=1,
                    scale=False,
                    svd_solver="full",
                ),
            ),
        ]
    )

    search = PiPLSSearchCV(
        estimator=pipeline,
        search_method="optimal",
        max_predictor_rank=4,
        cv=3,
        refit=False,
        n_jobs=1,
    ).fit(X, Y)

    assert search.max_predictor_rank_ == 2
    assert np.max(search.cv_results_["predictor_rank"]) == 2


def test_explicit_rank_above_fold_numerical_limit_is_rejected_before_scoring() -> None:
    X, Y = _rank_two_data()
    score_calls = 0

    def counting_scorer(
        estimator: object,
        X_validation: object,
        y_validation: object,
    ) -> float:
        nonlocal score_calls
        del estimator, X_validation, y_validation
        score_calls += 1
        return 0.0

    message = r"predictor_rank_values values must lie in \[1, 2\]"
    with pytest.raises(ValueError, match=message):
        PiPLSSearchCV(
            predictor_rank_values=[3],
            max_predictor_rank=5,
            search_method="optimal",
            cv=3,
            scoring=counting_scorer,
            refit=False,
            n_jobs=1,
        ).fit(X, Y)

    assert score_calls == 0


def test_no_positive_fold_numerical_rank_fails_transactionally() -> None:
    X, Y = _data(24)
    search = PiPLSSearchCV(
        n_components_values=[1],
        predictor_rank_values=[1],
        max_predictor_rank=1,
        cv=3,
        refit=False,
        n_jobs=1,
    ).fit(X, Y)

    with pytest.raises(ValueError, match="No positive predictor rank is numerically feasible"):
        search.fit(np.ones_like(X), Y)

    for name in (
        "max_predictor_rank_",
        "cv_results_",
        "component_path_",
        "best_params_",
    ):
        assert not hasattr(search, name)


def test_all_component_sentinel_matches_explicit_complete_range() -> None:
    X, Y = _data()
    all_search = PiPLSSearchCV(
        n_components_values="all",
        predictor_rank_values=[1, 2, 3],
        max_predictor_rank=3,
        search_method="optimal",
        cv=3,
        refit=False,
    ).fit(X, Y)
    explicit_search = PiPLSSearchCV(
        n_components_values=[1, 2, 3],
        predictor_rank_values=[1, 2, 3],
        max_predictor_rank=3,
        search_method="optimal",
        cv=3,
        refit=False,
    ).fit(X, Y)

    np.testing.assert_array_equal(
        all_search.component_path_.n_components,
        np.array([1, 2, 3]),
    )
    np.testing.assert_allclose(
        all_search.cv_results_["mean_test_score"],
        explicit_search.cv_results_["mean_test_score"],
    )


def test_default_scorer_name_resolves_to_the_public_callable() -> None:
    X, Y = _data()
    search = PiPLSSearchCV(
        n_components_values=[1],
        predictor_rank_values=[1],
        cv=3,
        refit=False,
    ).fit(X, Y)

    assert search.scorer_ is neg_response_standardized_mse
    np.testing.assert_allclose(
        search.cv_results_["mean_test_score"],
        -search.cv_results_["mean_response_standardized_mse"],
    )


def test_selected_estimator_is_refitted_and_delegates_prediction() -> None:
    X, Y = _data()
    search = PiPLSSearchCV(
        n_components_values=[1, 2],
        predictor_rank_values=[1, 2, 3],
        cv=3,
        refit=True,
        n_jobs=1,
    ).fit(X, Y)

    assert isinstance(search.selected_estimator_, PiPLSRegression)
    assert search.selected_pipls_ is search.selected_estimator_
    assert search.selected_params_ == search.best_params_
    assert search.selected_result_.n_components == search.best_n_components_
    assert search.selected_result_.predictor_rank == search.best_predictor_rank_
    assert search.selected_estimator_.n_components == search.best_n_components_
    assert search.selected_estimator_.predictor_rank == search.best_predictor_rank_
    np.testing.assert_allclose(search.predict(X), search.selected_estimator_.predict(X))
    assert search.score(X, Y) == pytest.approx(search.selected_estimator_.score(X, Y))


def test_one_standard_error_selection_refits_the_declared_path_row() -> None:
    X, Y = _one_standard_error_data()
    search = PiPLSSearchCV(
        n_components_values=[1, 2, 3],
        predictor_rank_values=[1, 2, 3, 4],
        search_method="optimal",
        selection_rule="one_standard_error",
        cv=4,
        refit=True,
        return_oof_predictions=True,
        n_jobs=1,
    ).fit(X, Y)

    expected = search.component_path_.one_standard_error_result()
    assert search.selected_result_ == expected
    assert search.selected_result_.n_components < search.best_n_components_
    assert search.selected_params_ == {
        "n_components": expected.n_components,
        "predictor_rank": expected.predictor_rank,
    }
    assert search.selected_estimator_.n_components == expected.n_components
    assert search.selected_estimator_.predictor_rank == expected.predictor_rank
    assert search.selected_pipls_ is search.selected_estimator_
    np.testing.assert_allclose(
        search.predict(X),
        search.selected_estimator_.predict(X),
    )
    assert search.validation_report_.selected_result is search.selected_result_
    assert search.validation_report_.n_components == expected.n_components
    assert search.validation_report_.predictor_rank == expected.predictor_rank
    assert search.validation_report_.mean_test_score == pytest.approx(
        expected.mean_test_score
    )
    assert search.validation_report_.oof_predictions is not None


def test_one_standard_error_selection_can_remain_selection_only() -> None:
    X, Y = _one_standard_error_data()
    search = PiPLSSearchCV(
        n_components_values=[1, 2, 3],
        predictor_rank_values=[1, 2, 3, 4],
        search_method="optimal",
        selection_rule="one_standard_error",
        cv=4,
        n_jobs=1,
    ).fit(X, Y)

    assert search.selected_result_ == search.component_path_.one_standard_error_result()
    assert not hasattr(search, "selected_estimator_")
    assert not hasattr(search, "selected_pipls_")
    assert not hasattr(search, "predict")


def test_one_standard_error_selection_requires_two_validation_splits() -> None:
    X, Y = _data()
    split = [(np.arange(24), np.arange(24, 36))]
    search = PiPLSSearchCV(
        n_components_values=[1],
        predictor_rank_values=[1],
        selection_rule="one_standard_error",
        cv=split,
    )

    with pytest.raises(ValueError, match="at least two validation splits"):
        search.fit(X, Y)

    with pytest.raises(NotFittedError):
        check_is_fitted(search)
    assert not hasattr(search, "selected_result_")


def test_default_selection_hides_refit_dependent_methods() -> None:
    X, Y = _data()
    search = PiPLSSearchCV(
        n_components_values=[1],
        predictor_rank_values=[1, 2],
        cv=3,
    )

    for method_name in (
        "predict",
        "transform",
        "fit_transform",
        "inverse_transform",
        "score",
        "get_feature_names_out",
        "set_output",
    ):
        assert not hasattr(search, method_name)

    search.fit(X, Y)
    for method_name in (
        "predict",
        "transform",
        "fit_transform",
        "inverse_transform",
        "score",
        "get_feature_names_out",
        "set_output",
    ):
        assert not hasattr(search, method_name)


def test_refit_false_clears_state_from_an_earlier_refitted_fit() -> None:
    X, Y = _data()
    search = PiPLSSearchCV(
        n_components_values=[1],
        predictor_rank_values=[1, 2],
        cv=3,
        refit=True,
    ).fit(X, Y)

    assert hasattr(search, "selected_estimator_")
    assert hasattr(search, "selected_pipls_")
    assert hasattr(search, "refit_time_")

    search.set_params(refit=False).fit(X, Y)

    assert not hasattr(search, "selected_estimator_")
    assert not hasattr(search, "selected_pipls_")
    assert not hasattr(search, "refit_time_")
    assert not hasattr(search, "predict")


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
    search = PiPLSSearchCV(
        estimator=pipeline,
        n_components_values=[1],
        predictor_rank_values=[1],
        cv=splits,
        refit=True,
        n_jobs=1,
    ).fit(X, Y)

    manual = pipeline.fit(X[:12], Y[:12])
    prediction = manual.predict(X[12:18])
    response_scale = np.std(Y[:12], axis=0, ddof=1)
    expected = np.mean(((Y[12:18] - prediction) / response_scale[None, :]) ** 2)

    assert search.cv_results_["split0_response_standardized_mse"][0] == pytest.approx(expected)
    assert isinstance(search.selected_estimator_, Pipeline)


def test_auto_path_skips_candidates_with_constant_scorer() -> None:
    rng = np.random.default_rng(20260717)
    X = rng.normal(size=(80, 20))
    Y = X @ rng.normal(size=(20, 3)) + 0.05 * rng.normal(size=(80, 3))

    def constant_scorer(estimator: object, X_validation: object, y_validation: object) -> float:
        del estimator, X_validation, y_validation
        return 1.0

    search = PiPLSSearchCV(
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
    assert search.cv_results_["predictor_rank"].size < 12
    assert not search.path_search_exhaustive_

    profile = search.predictor_rank_profile(1)
    np.testing.assert_array_equal(
        profile.predictor_rank,
        np.sort(search.cv_results_["predictor_rank"]),
    )
    assert profile.predictor_rank.size == search.cv_results_["predictor_rank"].size
    assert profile.selected.predictor_rank == 1


def test_rank_test_score_one_matches_the_best_score_tolerance_group() -> None:
    X, Y = _data()
    score_by_rank = {
        1: 1.0,
        2: 1.0 - 0.75e-12,
        3: 1.0 - 1.50e-12,
    }

    def chained_scores(
        estimator: object,
        X_validation: object,
        y_validation: object,
    ) -> float:
        del X_validation, y_validation
        predictor_rank = int(estimator.predictor_rank)
        return score_by_rank[predictor_rank]

    search = PiPLSSearchCV(
        n_components_values=[1],
        predictor_rank_values=[1, 2, 3],
        max_predictor_rank=3,
        search_method="optimal",
        scoring=chained_scores,
        cv=3,
        refit=False,
        n_jobs=1,
    ).fit(X, Y)

    np.testing.assert_array_equal(
        search.cv_results_["predictor_rank"],
        np.array([1, 2, 3]),
    )
    np.testing.assert_array_equal(
        search.cv_results_["rank_test_score"],
        np.array([1, 1, 3]),
    )
    assert search.best_predictor_rank_ == 1
    assert search.best_index_ == 0


def test_global_tie_breaking_prefers_lower_components_then_rank() -> None:
    X, Y = _data()

    def constant_scorer(estimator: object, X_validation: object, y_validation: object) -> float:
        del estimator, X_validation, y_validation
        return 1.0

    search = PiPLSSearchCV(
        n_components_values=[1, 2],
        predictor_rank_values=[1, 2, 3],
        scoring=constant_scorer,
        cv=3,
    ).fit(X, Y)

    assert search.best_n_components_ == 1
    assert search.best_predictor_rank_ == 1
    np.testing.assert_array_equal(search.component_path_.predictor_rank, np.array([1, 2]))
    np.testing.assert_allclose(search.component_path_.mean_test_score, np.array([1.0, 1.0]))
    assert not np.allclose(
        search.component_path_.cv_mse_mean,
        -search.component_path_.mean_test_score,
    )

    profile = search.predictor_rank_profile(2)
    assert profile.selected.predictor_rank == 2
    assert np.all(profile.mean_test_score == 1.0)
    assert not np.allclose(profile.cv_mse_mean, -profile.mean_test_score)


def test_explicit_max_predictor_rank_bypasses_rule_bound() -> None:
    X, Y = _data()
    search = PiPLSSearchCV(
        n_components_values=[1],
        predictor_rank_values=[1, 2, 3],
        max_predictor_rank=3,
        samples_per_predictor_rank=100,
        cv=3,
    ).fit(X, Y)

    assert search.max_predictor_rank_ == 3
    np.testing.assert_array_equal(
        search.predictor_rank_profile(1).predictor_rank,
        np.array([1, 2, 3]),
    )


def test_low_samples_per_predictor_rank_warns() -> None:
    X, Y = _data()
    with pytest.warns(StatisticalSupportWarning, match="statistical support"):
        PiPLSSearchCV(
            n_components_values=[1],
            predictor_rank_values=[1],
            samples_per_predictor_rank=4,
            cv=3,
        ).fit(X, Y)


def test_oof_generation_does_not_rescore_the_selected_candidate() -> None:
    X, Y = _data(18)
    calls = 0

    def counting_scorer(
        estimator: object,
        X_validation: object,
        y_validation: object,
    ) -> float:
        nonlocal calls
        del estimator, X_validation, y_validation
        calls += 1
        return 0.0

    PiPLSSearchCV(
        n_components_values=[1],
        predictor_rank_values=[1],
        max_predictor_rank=1,
        cv=3,
        scoring=counting_scorer,
        refit=False,
        return_oof_predictions=True,
        n_jobs=1,
    ).fit(X, Y)

    assert calls == 3


def test_path_suppresses_direct_fit_support_warning_through_oof_and_refit() -> None:
    X, Y = _data(12)

    with warnings.catch_warnings():
        warnings.simplefilter("error", StatisticalSupportWarning)
        search = PiPLSSearchCV(
            n_components_values=[1],
            predictor_rank_values=[4],
            max_predictor_rank=4,
            cv=3,
            refit=True,
            return_oof_predictions=True,
            n_jobs=1,
        ).fit(X, Y)

    assert search.best_predictor_rank_ == 4
    assert search.selected_pipls_.predictor_rank_ == 4
    report = search.validation_report_
    assert report.oof_prediction_counts is not None
    np.testing.assert_array_equal(report.oof_prediction_counts, np.ones(X.shape[0]))


def test_path_does_not_suppress_unrelated_estimator_warnings() -> None:
    X, Y = _data(18)
    pipeline = Pipeline(
        [
            ("warning", _WarningTransformer()),
            ("regression", PiPLSRegression(n_components=1, predictor_rank=1)),
        ]
    )

    with pytest.warns(RuntimeWarning, match="unrelated path warning"):
        PiPLSSearchCV(
            estimator=pipeline,
            n_components_values=[1],
            predictor_rank_values=[1],
            max_predictor_rank=1,
            cv=2,
            refit=False,
            n_jobs=1,
        ).fit(X, Y)


def test_path_clones_the_fixed_estimator_template_without_mutating_it() -> None:
    X, Y = _data()
    template = PiPLSRegression(n_components=2, predictor_rank=2)

    search = PiPLSSearchCV(
        estimator=template,
        n_components_values=[1],
        predictor_rank_values=[3],
        max_predictor_rank=3,
        cv=3,
        refit=True,
        n_jobs=1,
    ).fit(X, Y)

    assert template.n_components == 2
    assert template.predictor_rank == 2
    assert not hasattr(template, "coef_")
    assert search.selected_pipls_.n_components == 1
    assert search.selected_pipls_.predictor_rank == 3


@pytest.mark.parametrize(
    ("keyword", "value", "message"),
    [
        ("search_method", "exhaustive", "search_method"),
        ("max_predictor_rank", 0, "max_predictor_rank"),
        ("n_components_values", [], "must not be empty"),
        ("n_components_values", None, 'must be "all"'),
        ("n_components_values", "everything", 'must be "all"'),
        ("scoring", "not_a_scorer", "Unknown scoring"),
        ("predictor_rank_values", [1.0], "positive integer"),
        ("predictor_rank_values", "maximum", "must be None"),
        ("n_jobs", 0, "must not be zero"),
        ("selection_rule", "smallest", "selection_rule"),
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
        PiPLSSearchCV(**kwargs).fit(X, Y)


def test_component_path_exposes_conditional_scores_and_cv_mse_summaries() -> None:
    X, Y = _data()
    search = PiPLSSearchCV(
        n_components_values=[1, 2],
        predictor_rank_values=[1, 2, 3, 4],
        search_method="optimal",
        cv=3,
        refit=False,
        n_jobs=1,
    ).fit(X, Y)

    path = search.component_path_
    assert isinstance(path, PiPLSComponentPath)
    np.testing.assert_array_equal(path.n_components, np.array([1, 2]))
    assert path.n_splits == 3
    assert path.predictor_rank_policy == "optimized"

    for row_index, h in enumerate((1, 2)):
        rank = int(path.predictor_rank[row_index])
        result_index = np.flatnonzero(
            (search.cv_results_["n_components"] == h)
            & (search.cv_results_["predictor_rank"] == rank)
        )
        assert result_index.size == 1
        index = int(result_index[0])
        assert path.mean_test_score[row_index] == pytest.approx(
            search.cv_results_["mean_test_score"][index]
        )
        assert path.cv_mse_mean[row_index] == pytest.approx(
            search.cv_results_["mean_response_standardized_mse"][index]
        )
        assert path.cv_mse_fold_sd[row_index] == pytest.approx(
            search.cv_results_["std_response_standardized_mse"][index]
        )
        assert path.cv_mse_standard_error[row_index] == pytest.approx(
            search.cv_results_["std_response_standardized_mse"][index]
            / np.sqrt(path.n_splits - 1)
        )

    selected = path.for_n_components(search.best_n_components_)
    assert selected.predictor_rank == search.best_predictor_rank_
    assert selected.mean_test_score == pytest.approx(search.best_score_)

    restored = pickle.loads(pickle.dumps(search))
    np.testing.assert_array_equal(restored.component_path_.n_components, path.n_components)
    assert not restored.component_path_.n_components.flags.writeable


def test_predictor_rank_profile_is_sorted_and_consistent_with_cv_results() -> None:
    X, Y = _data()
    search = PiPLSSearchCV(
        n_components_values=[1, 2],
        predictor_rank_values=[1, 2, 3, 4],
        search_method="optimal",
        cv=3,
        refit=False,
        n_jobs=1,
    ).fit(X, Y)

    profile = search.predictor_rank_profile(2)

    assert isinstance(profile, PiPLSPredictorRankProfile)
    assert profile.n_components == 2
    assert profile.n_splits == 3
    np.testing.assert_allclose(
        profile.cv_mse_standard_error,
        profile.cv_mse_fold_sd / np.sqrt(profile.n_splits - 1),
    )
    assert not profile.cv_mse_standard_error.flags.writeable
    np.testing.assert_array_equal(profile.predictor_rank, np.array([2, 3, 4]))
    assert all(
        not array.flags.writeable
        for array in (
            profile.predictor_rank,
            profile.mean_test_score,
            profile.cv_mse_mean,
            profile.cv_mse_fold_sd,
        )
    )
    rows = np.flatnonzero(search.cv_results_["n_components"] == 2)
    order = np.argsort(search.cv_results_["predictor_rank"][rows])
    indices = rows[order]
    np.testing.assert_allclose(
        profile.mean_test_score,
        search.cv_results_["mean_test_score"][indices],
    )
    np.testing.assert_allclose(
        profile.cv_mse_mean,
        search.cv_results_["mean_response_standardized_mse"][indices],
    )
    np.testing.assert_allclose(
        profile.cv_mse_fold_sd,
        search.cv_results_["std_response_standardized_mse"][indices],
    )
    assert profile.selected == search.component_path_.for_n_components(2)


def test_predictor_rank_profile_requires_fitted_evaluated_component_count() -> None:
    search = PiPLSSearchCV(refit=False)

    with pytest.raises(NotFittedError):
        search.predictor_rank_profile(1)

    X, Y = _data()
    search.set_params(n_components_values=[1, 2], cv=3).fit(X, Y)
    with pytest.raises(ValueError, match="was not evaluated"):
        search.predictor_rank_profile(3)
    with pytest.raises(ValueError, match="must be an integer"):
        search.predictor_rank_profile(2.0)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("predictor_rank_values", "expected_policy"),
    [
        (None, "optimized"),
        ([3], "fixed"),
        ("max", "maximum"),
        ([2, 3, 4], "optimized"),
    ],
)
def test_component_path_records_predictor_rank_policy(
    predictor_rank_values: object,
    expected_policy: str,
) -> None:
    X, Y = _data()
    search = PiPLSSearchCV(
        n_components_values=[1, 2],
        predictor_rank_values=predictor_rank_values,
        cv=3,
        refit=False,
        n_jobs=1,
    ).fit(X, Y)

    assert search.component_path_.predictor_rank_policy == expected_policy
    if expected_policy == "fixed":
        np.testing.assert_array_equal(
            search.component_path_.predictor_rank,
            np.array([3, 3]),
        )
    if expected_policy == "maximum":
        np.testing.assert_array_equal(
            search.component_path_.predictor_rank,
            np.full(2, search.max_predictor_rank_),
        )

    profile = search.predictor_rank_profile(2)
    assert profile.selected.predictor_rank == search.component_path_.for_n_components(
        2
    ).predictor_rank
    if expected_policy in {"fixed", "maximum"}:
        assert profile.predictor_rank.size == 1


def test_fixed_predictor_rank_must_support_every_component_count() -> None:
    X, Y = _data()

    with pytest.raises(ValueError, match="Every n_components value"):
        PiPLSSearchCV(
            n_components_values=[1, 2, 3],
            predictor_rank_values=[2],
            cv=3,
        ).fit(X, Y)
