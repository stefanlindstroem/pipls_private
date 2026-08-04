from __future__ import annotations

import pickle
import warnings
from dataclasses import FrozenInstanceError

import numpy as np
import pytest
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.exceptions import NotFittedError
from sklearn.model_selection import RepeatedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from pipls import PiPLSRegression, PiPLSSearchCV, PredictorRankSupportWarning
from pipls.component_path import (
    PiPLSComponentPath,
    PiPLSPredictorRankProfile,
    PiPLSSelection,
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


def _search_with_component_path(path: PiPLSComponentPath) -> PiPLSSearchCV:
    """Return minimal fitted-state evidence for exact selection tests."""

    search = PiPLSSearchCV()
    search.cv_results_ = {
        "n_components": path.n_components,
        "predictor_rank": path.predictor_rank,
        "mean_test_score": path.mean_test_score,
    }
    search.component_path_ = path
    return search


def test_optimal_path_evaluates_complete_triangular_grid() -> None:
    X, Y = _data()
    search = PiPLSSearchCV(
        samples_per_predictor_rank=8,
        cv=3,
        n_jobs=1,
    ).fit(X, Y)

    assert search.max_predictor_rank_ == 5
    assert search.search_is_exhaustive_
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
        n_jobs=1,
    ).fit(X, Y)

    with pytest.raises(ValueError, match="No positive predictor rank is numerically feasible"):
        search.fit(np.ones_like(X), Y)

    for name in (
        "max_predictor_rank_",
        "cv_results_",
        "component_path_",
        "best_index_",
        "best_score_",
        "best_n_components_",
        "best_predictor_rank_",
        "best_params_",
    ):
        assert not hasattr(search, name)


def test_fit_exposes_no_global_best_attributes() -> None:
    X, Y = _data()
    search = PiPLSSearchCV(
        n_components_values=[1, 2],
        predictor_rank_values=[1, 2, 3],
        search_method="optimal",
        cv=3,
        n_jobs=1,
    ).fit(X, Y)

    best = search.select(rule="best_score")
    assert best.n_components in {1, 2}
    assert best.predictor_rank in {1, 2, 3}
    for name in (
        "best_index_",
        "best_score_",
        "best_n_components_",
        "best_predictor_rank_",
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
    ).fit(X, Y)
    explicit_search = PiPLSSearchCV(
        n_components_values=[1, 2, 3],
        predictor_rank_values=[1, 2, 3],
        max_predictor_rank=3,
        search_method="optimal",
        cv=3,
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
    ).fit(X, Y)

    assert search.scorer_ is neg_response_standardized_mse
    np.testing.assert_allclose(
        search.cv_results_["mean_test_score"],
        -search.cv_results_["mean_response_standardized_mse"],
    )


def test_post_fit_select_returns_immutable_stored_results_without_mutation() -> None:
    X, Y = _one_standard_error_data()
    search = PiPLSSearchCV(
        n_components_values=[1, 2, 3],
        predictor_rank_values=[1, 2, 3, 4],
        search_method="optimal",
        cv=4,
        n_jobs=1,
    ).fit(X, Y)
    state_before = dict(search.__dict__)

    selected = search.select(n_components=np.int64(2))
    assert selected == search.predictor_rank_profile(2).selection
    assert selected.rule is None
    assert selected.reference_minimum is None
    assert selected.one_standard_error_threshold is None

    best = search.select(rule="best_score")
    best_row = search.select(n_components=best.n_components)
    assert best.rule == "best_score"
    assert best.reference_minimum is None
    assert best.one_standard_error_threshold is None
    assert best.n_components == best_row.n_components
    assert best.predictor_rank == best_row.predictor_rank
    assert best.cv_mse_mean == best_row.cv_mse_mean

    path = search.component_path_
    minimum_index = int(np.argmin(path.cv_mse_mean))
    minimum = search.select(rule="minimum_cv_mse")
    assert minimum.n_components == int(path.n_components[minimum_index])
    assert minimum.predictor_rank == int(path.predictor_rank[minimum_index])
    assert minimum.cv_mse_mean == path.cv_mse_mean[minimum_index]
    assert minimum.rule == "minimum_cv_mse"
    assert minimum.reference_minimum is None
    assert minimum.one_standard_error_threshold is None

    threshold = minimum.cv_mse_mean + minimum.cv_mse_standard_error
    eligible_index = int(np.flatnonzero(path.cv_mse_mean <= threshold)[0])
    one_se = search.select(rule="one_standard_error")
    assert one_se.n_components == int(path.n_components[eligible_index])
    assert one_se.predictor_rank == int(path.predictor_rank[eligible_index])
    assert one_se.rule == "one_standard_error"
    assert one_se.reference_minimum == minimum
    assert one_se.one_standard_error_threshold == threshold

    with pytest.raises(FrozenInstanceError):
        selected.n_components = 99  # type: ignore[misc]

    assert search.__dict__.keys() == state_before.keys()
    for name, value in state_before.items():
        assert search.__dict__[name] is value


def test_post_fit_select_validates_selection_input_and_fitted_state() -> None:
    X, Y = _data()
    unfitted = PiPLSSearchCV()

    with pytest.raises(NotFittedError):
        unfitted.select(rule="best_score")

    search = PiPLSSearchCV(
        n_components_values=[1, 2],
        predictor_rank_values=[1, 2],
        cv=3,
    ).fit(X, Y)
    with pytest.raises(ValueError, match="Exactly one of rule and n_components"):
        search.select()
    with pytest.raises(ValueError, match="Exactly one of rule and n_components"):
        search.select(rule="best_score", n_components=1)
    with pytest.raises(ValueError, match="rule must be"):
        search.select(rule="smallest")  # type: ignore[arg-type]
    with pytest.raises(
        ValueError,
        match=r"n_components=3 was not evaluated.*\[1, 2\]",
    ):
        search.select(n_components=3)
    with pytest.raises(ValueError, match="must be an integer"):
        search.select(n_components=2.0)  # type: ignore[arg-type]


def test_select_component_count_returns_the_complete_exact_stored_row() -> None:
    path = PiPLSComponentPath(
        n_components=[1, 2, 4],
        predictor_rank=[3, 4, 5],
        predictor_rank_policy="optimized",
        mean_test_score=[-0.8, -0.5, -0.45],
        cv_mse_mean=[0.8, 0.5, 0.45],
        cv_mse_std=[0.1, 0.08, 0.07],
        n_splits=5,
    )
    search = _search_with_component_path(path)

    selected = search.select(n_components=np.int64(2))

    assert selected == PiPLSSelection(
        n_components=2,
        predictor_rank=4,
        predictor_rank_policy="optimized",
        mean_test_score=-0.5,
        cv_mse_mean=0.5,
        cv_mse_std=0.08,
        n_splits=5,
    )
    assert type(selected.n_components) is int
    assert type(selected.predictor_rank) is int
    assert type(selected.mean_test_score) is float


def test_select_minimum_cv_mse_returns_first_exact_stored_tie() -> None:
    path = PiPLSComponentPath(
        n_components=[1, 2, 4],
        predictor_rank=[2, 4, 6],
        predictor_rank_policy="optimized",
        mean_test_score=[-0.5, -0.4, -0.4],
        cv_mse_mean=[0.5, 0.4, 0.4],
        cv_mse_std=[0.1, 0.08, 0.07],
        n_splits=5,
    )
    search = _search_with_component_path(path)

    selected = search.select(rule="minimum_cv_mse")

    assert selected == PiPLSSelection(
        n_components=2,
        predictor_rank=4,
        predictor_rank_policy="optimized",
        mean_test_score=-0.4,
        cv_mse_mean=0.4,
        cv_mse_std=0.08,
        n_splits=5,
        rule="minimum_cv_mse",
    )


def test_select_one_standard_error_uses_the_minimum_rows_standard_error() -> None:
    path = PiPLSComponentPath(
        n_components=[1, 2, 3, 4],
        predictor_rank=[2, 3, 5, 6],
        predictor_rank_policy="optimized",
        mean_test_score=[-0.48, -0.45, -0.40, -0.42],
        cv_mse_mean=[0.48, 0.45, 0.40, 0.42],
        cv_mse_std=[1.0, 0.4, 0.08, 0.2],
        n_splits=5,
    )
    search = _search_with_component_path(path)

    selected = search.select(rule="one_standard_error")

    assert selected.n_components == 3
    assert selected.predictor_rank == 5
    assert selected.rule == "one_standard_error"
    assert selected.reference_minimum == search.select(rule="minimum_cv_mse")
    assert selected.one_standard_error_threshold == pytest.approx(0.44)


def test_select_one_standard_error_returns_smallest_eligible_count() -> None:
    path = PiPLSComponentPath(
        n_components=[1, 2, 4],
        predictor_rank=[2, 3, 5],
        predictor_rank_policy="optimized",
        mean_test_score=[-0.50, -0.44, -0.40],
        cv_mse_mean=[0.50, 0.44, 0.40],
        cv_mse_std=[0.1, 0.1, 0.10],
        n_splits=5,
    )
    search = _search_with_component_path(path)

    selected = search.select(rule="one_standard_error")

    assert selected.n_components == 2
    assert selected.predictor_rank == 3
    assert selected.reference_minimum == search.select(rule="minimum_cv_mse")


def test_select_one_standard_error_uses_no_extra_tolerance() -> None:
    minimum = 0.4
    reference_standard_error = 0.04
    threshold = minimum + reference_standard_error
    just_above_threshold = np.nextafter(threshold, np.inf)
    path = PiPLSComponentPath(
        n_components=[1, 2, 3],
        predictor_rank=[2, 3, 4],
        predictor_rank_policy="optimized",
        mean_test_score=[-0.5, -just_above_threshold, -minimum],
        cv_mse_mean=[0.5, just_above_threshold, minimum],
        cv_mse_std=[0.1, 0.1, 2.0 * reference_standard_error],
        n_splits=5,
    )
    search = _search_with_component_path(path)

    selected = search.select(rule="one_standard_error")

    assert selected.n_components == 3
    assert selected.one_standard_error_threshold == threshold


def test_select_rules_handle_split_and_finite_threshold_edges() -> None:
    one_split = _search_with_component_path(
        PiPLSComponentPath(
            n_components=[1, 2],
            predictor_rank=[2, 3],
            predictor_rank_policy="optimized",
            mean_test_score=[-0.4, -0.5],
            cv_mse_mean=[0.4, 0.5],
            cv_mse_std=[0.0, 0.1],
            n_splits=1,
        )
    )
    assert one_split.select(rule="minimum_cv_mse").n_components == 1
    with pytest.raises(ValueError, match="requires at least two"):
        one_split.select(rule="one_standard_error")

    overflowing = _search_with_component_path(
        PiPLSComponentPath(
            n_components=[1],
            predictor_rank=[1],
            predictor_rank_policy="fixed",
            mean_test_score=[-1.0e308],
            cv_mse_mean=[1.0e308],
            cv_mse_std=[1.0e308],
            n_splits=2,
        )
    )
    with pytest.raises(ValueError, match="threshold must be finite"):
        overflowing.select(rule="one_standard_error")


def test_post_fit_refit_returns_fitted_direct_model_for_best_score() -> None:
    X, Y = _data()
    search = PiPLSSearchCV(
        n_components_values=[1, 2],
        predictor_rank_values=[1, 2, 3],
        cv=3,
        n_jobs=1,
    ).fit(X, Y)

    model = search.refit(X, Y, rule="best_score")

    assert isinstance(model, PiPLSRegression)
    assert model.n_components == model.selection_.n_components
    assert model.predictor_rank == model.selection_.predictor_rank
    assert hasattr(model, "coef_")
    assert model.predict(X).shape == Y.shape
    assert model.selection_ == search.select(rule="best_score")
    assert not hasattr(search, "selected_estimator_")
    assert not hasattr(search, "selected_pipls_")


def test_post_fit_refit_supports_manual_component_selection() -> None:
    X, Y = _data()
    search = PiPLSSearchCV(
        n_components_values=[1, 2, 3],
        predictor_rank_values=[1, 2, 3, 4],
        search_method="optimal",
        cv=4,
        n_jobs=1,
    ).fit(X, Y)
    expected = search.select(n_components=2)

    model = search.refit(X, Y, n_components=2)

    assert isinstance(model, PiPLSRegression)
    assert model.n_components == expected.n_components
    assert model.predictor_rank == expected.predictor_rank
    assert model.selection_ == expected
    assert model.selection_.rule is None


def test_post_fit_refit_supports_component_path_rules() -> None:
    X, Y = _one_standard_error_data()
    search = PiPLSSearchCV(
        n_components_values=[1, 2, 3],
        predictor_rank_values=[1, 2, 3, 4],
        search_method="optimal",
        cv=4,
        n_jobs=1,
    ).fit(X, Y)

    expected_by_rule = {
        rule: search.select(rule=rule)
        for rule in ("best_score", "minimum_cv_mse", "one_standard_error")
    }
    for rule, expected in expected_by_rule.items():
        model = search.refit(X, Y, rule=rule)  # type: ignore[arg-type]
        assert isinstance(model, PiPLSRegression)
        assert model.n_components == expected.n_components
        assert model.predictor_rank == expected.predictor_rank
        assert model.selection_ == expected
        assert model.selection_.rule == rule


def test_best_score_and_minimum_cv_mse_rules_can_select_different_models() -> None:
    X, Y = _data()
    search_kwargs = {
        "n_components_values": [1, 2, 3],
        "predictor_rank_values": [3],
        "max_predictor_rank": 3,
        "search_method": "optimal",
        "cv": 3,
        "n_jobs": 1,
    }
    baseline = PiPLSSearchCV(**search_kwargs).fit(X, Y)
    minimum_components = baseline.select(rule="minimum_cv_mse").n_components
    favored_components = 3 if minimum_components != 3 else 1

    def component_scorer(
        estimator: object,
        X_validation: object,
        y_validation: object,
    ) -> float:
        del X_validation, y_validation
        return float(
            estimator.n_components == favored_components
        )

    search = PiPLSSearchCV(
        **search_kwargs,
        scoring=component_scorer,
    ).fit(X, Y)
    best_selected = search.select(rule="best_score")
    minimum_selected = search.select(rule="minimum_cv_mse")
    best_model = search.refit(X, Y, rule="best_score")
    minimum_model = search.refit(X, Y, rule="minimum_cv_mse")

    assert best_selected.n_components == favored_components
    assert minimum_selected.n_components == minimum_components
    assert best_model.n_components == best_selected.n_components
    assert minimum_model.n_components == minimum_selected.n_components
    assert best_model.n_components != minimum_model.n_components


def test_post_fit_operations_create_no_selected_search_state() -> None:
    X, Y = _one_standard_error_data()
    search = PiPLSSearchCV(
        n_components_values=[1, 2, 3],
        predictor_rank_values=[1, 2, 3, 4],
        search_method="optimal",
        cv=4,
        n_jobs=1,
    ).fit(X, Y)

    expected = search.select(rule="one_standard_error")
    model = search.refit(X, Y, rule="one_standard_error")
    report = search.oof_report(X, Y, selection=model.selection_)

    assert report.selection == expected
    assert report.selection.rule == "one_standard_error"
    assert report.selection.reference_minimum == search.select(
        rule="minimum_cv_mse"
    )
    assert report.selection.one_standard_error_threshold is not None
    assert model.n_components == expected.n_components
    assert model.predictor_rank == expected.predictor_rank
    assert model.selection_ == expected
    assert model.selection_ == report.selection
    for name in ("selected_params_", "oof_report_"):
        assert not hasattr(search, name)


def test_post_fit_refit_requires_exactly_one_selection_input() -> None:
    X, Y = _data()
    search = PiPLSSearchCV(
        n_components_values=[1, 2],
        predictor_rank_values=[1, 2],
        cv=3,
    ).fit(X, Y)

    with pytest.raises(ValueError, match="Exactly one of rule and n_components"):
        search.refit(X, Y)
    with pytest.raises(ValueError, match="Exactly one of rule and n_components"):
        search.refit(X, Y, rule="best_score", n_components=1)
    with pytest.raises(ValueError, match="rule must be"):
        search.refit(X, Y, rule="smallest")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="was not evaluated"):
        search.refit(X, Y, n_components=3)


def test_post_fit_refit_requires_a_fitted_search() -> None:
    X, Y = _data()
    search = PiPLSSearchCV()

    with pytest.raises(NotFittedError):
        search.refit(X, Y, rule="best_score")


def test_one_standard_error_operations_require_two_validation_splits() -> None:
    X, Y = _data()
    split = [(np.arange(24), np.arange(24, 36))]
    search = PiPLSSearchCV(
        n_components_values=[1],
        predictor_rank_values=[1],
        cv=split,
    ).fit(X, Y)

    with pytest.raises(ValueError, match="at least two validation splits"):
        search.select(rule="one_standard_error")
    with pytest.raises(ValueError, match="at least two validation splits"):
        search.refit(X, Y, rule="one_standard_error")


def test_post_fit_refit_does_not_mutate_search_state() -> None:
    X, Y = _data()
    search = PiPLSSearchCV(
        n_components_values=[1, 2],
        predictor_rank_values=[1, 2, 3],
        cv=3,
    ).fit(X, Y)
    before = pickle.dumps(search)

    search.refit(X, Y, n_components=1)

    assert pickle.dumps(search) == before


def test_refitted_model_selection_is_pickle_stable_and_not_cloned() -> None:
    X, Y = _one_standard_error_data()
    search = PiPLSSearchCV(
        n_components_values=[1, 2, 3],
        predictor_rank_values=[1, 2, 3, 4],
        search_method="optimal",
        cv=4,
        n_jobs=1,
    ).fit(X, Y)

    model = search.refit(X, Y, rule="one_standard_error")
    restored = pickle.loads(pickle.dumps(model))
    cloned = clone(model)

    assert restored.selection_ == model.selection_
    assert restored.selection_.reference_minimum == search.select(
        rule="minimum_cv_mse"
    )
    assert not hasattr(cloned, "selection_")


def test_refit_attaches_selection_only_after_successful_fit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    X, Y = _data()
    search = PiPLSSearchCV(
        n_components_values=[1],
        predictor_rank_values=[1],
        cv=3,
    ).fit(X, Y)

    def fail_fit(estimator: object, X_fit: object, y_fit: object) -> None:
        del X_fit, y_fit
        assert not hasattr(estimator, "selection_")
        raise RuntimeError("intentional refit failure")

    monkeypatch.setattr("pipls.search._fit_path_estimator", fail_fit)

    with pytest.raises(RuntimeError, match="intentional refit failure"):
        search.refit(X, Y, n_components=1)


def test_failed_post_fit_refit_leaves_search_state_unchanged() -> None:
    X, Y = _data()
    search = PiPLSSearchCV(
        n_components_values=[1],
        predictor_rank_values=[1],
        cv=3,
    ).fit(X, Y)
    before = pickle.dumps(search)

    with pytest.raises(ValueError):
        search.refit(X[:-1], Y, n_components=1)

    assert pickle.dumps(search) == before


def test_search_exposes_evidence_and_refit_but_no_model_delegation() -> None:
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
    assert hasattr(search, "refit")
    assert hasattr(search, "oof_report")
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
        n_jobs=1,
    ).fit(X, Y)

    manual = pipeline.fit(X[:12], Y[:12])
    prediction = manual.predict(X[12:18])
    response_scale = np.std(Y[:12], axis=0, ddof=1)
    expected = np.mean(((Y[12:18] - prediction) / response_scale[None, :]) ** 2)

    assert search.cv_results_["split0_response_standardized_mse"][0] == pytest.approx(expected)
    model = search.refit(X, Y, rule="best_score")
    assert isinstance(model, Pipeline)
    assert model is not pipeline


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

    best = search.select(rule="best_score")
    assert best.n_components == 1
    assert best.predictor_rank == 1
    assert search.cv_results_["predictor_rank"].size < 12
    assert not search.search_is_exhaustive_

    profile = search.predictor_rank_profile(1)
    np.testing.assert_array_equal(
        profile.predictor_rank,
        np.sort(search.cv_results_["predictor_rank"]),
    )
    assert profile.predictor_rank.size == search.cv_results_["predictor_rank"].size
    assert profile.selection.predictor_rank == 1


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
    best = search.select(rule="best_score")
    assert best.predictor_rank == 1
    assert best.mean_test_score == pytest.approx(1.0)


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

    best = search.select(rule="best_score")
    assert best.n_components == 1
    assert best.predictor_rank == 1
    np.testing.assert_array_equal(search.component_path_.predictor_rank, np.array([1, 2]))
    np.testing.assert_allclose(search.component_path_.mean_test_score, np.array([1.0, 1.0]))
    assert not np.allclose(
        search.component_path_.cv_mse_mean,
        -search.component_path_.mean_test_score,
    )

    profile = search.predictor_rank_profile(2)
    assert profile.selection.predictor_rank == 2
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
    assert not hasattr(search, "cv_n_train_min_")
    np.testing.assert_array_equal(
        search.predictor_rank_profile(1).predictor_rank,
        np.array([1, 2, 3]),
    )


def test_low_samples_per_predictor_rank_warns() -> None:
    X, Y = _data()
    with pytest.warns(PredictorRankSupportWarning, match="statistical support"):
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

    search = PiPLSSearchCV(
        n_components_values=[1],
        predictor_rank_values=[1],
        max_predictor_rank=1,
        cv=3,
        scoring=counting_scorer,
        n_jobs=1,
    ).fit(X, Y)
    selection = search.select(n_components=1)
    search.oof_report(X, Y, selection=selection)

    assert calls == 3


def test_path_suppresses_direct_fit_support_warning_through_oof_and_post_fit_refit() -> None:
    X, Y = _data(12)

    with warnings.catch_warnings():
        warnings.simplefilter("error", PredictorRankSupportWarning)
        search = PiPLSSearchCV(
            n_components_values=[1],
            predictor_rank_values=[4],
            max_predictor_rank=4,
            cv=3,
            n_jobs=1,
        ).fit(X, Y)
        model = search.refit(X, Y, rule="best_score")
        report = search.oof_report(X, Y, selection=model.selection_)

    assert model.selection_.predictor_rank == 4
    assert isinstance(model, PiPLSRegression)
    assert model.predictor_rank == 4
    assert not hasattr(model, "predictor_rank_")
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
        search = PiPLSSearchCV(
            estimator=pipeline,
            n_components_values=[1],
            predictor_rank_values=[1],
            max_predictor_rank=1,
            cv=2,
            n_jobs=1,
        ).fit(X, Y)
    selection = search.select(n_components=1)
    with pytest.warns(RuntimeWarning, match="unrelated path warning"):
        search.oof_report(X, Y, selection=selection)
    with pytest.warns(RuntimeWarning, match="unrelated path warning"):
        search.refit(X, Y, n_components=1)


def test_path_clones_the_fixed_estimator_template_without_mutating_it() -> None:
    X, Y = _data()
    template = PiPLSRegression(n_components=2, predictor_rank=2)

    search = PiPLSSearchCV(
        estimator=template,
        n_components_values=[1],
        predictor_rank_values=[3],
        max_predictor_rank=3,
        cv=3,
        n_jobs=1,
    ).fit(X, Y)
    model = search.refit(X, Y, n_components=1)

    assert template.n_components == 2
    assert template.predictor_rank == 2
    assert not hasattr(template, "coef_")
    assert model.n_components == 1
    assert model.predictor_rank == 3


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
        assert path.cv_mse_std[row_index] == pytest.approx(
            search.cv_results_["std_response_standardized_mse"][index]
        )
        assert path.cv_mse_standard_error[row_index] == pytest.approx(
            search.cv_results_["std_response_standardized_mse"][index]
            / np.sqrt(path.n_splits - 1)
        )

    selected = search.select(rule="best_score")
    best_index = np.flatnonzero(
        (search.cv_results_["n_components"] == selected.n_components)
        & (search.cv_results_["predictor_rank"] == selected.predictor_rank)
    )
    assert best_index.size == 1
    assert selected.mean_test_score == pytest.approx(
        search.cv_results_["mean_test_score"][int(best_index[0])]
    )

    restored = pickle.loads(pickle.dumps(search))
    np.testing.assert_array_equal(restored.component_path_.n_components, path.n_components)
    assert not restored.component_path_.n_components.flags.writeable


@pytest.mark.parametrize(
    ("cv", "expected_n_splits"),
    [
        pytest.param(
            RepeatedKFold(n_splits=3, n_repeats=2, random_state=0),
            6,
            id="repeated-k-fold",
        ),
        pytest.param(
            [(np.arange(6, 18), np.arange(0, 6))],
            1,
            id="one-split",
        ),
        pytest.param(
            [
                (np.arange(3, 18), np.arange(0, 3)),
                (np.concatenate((np.arange(0, 3), np.arange(8, 18))), np.arange(3, 8)),
                (np.arange(0, 8), np.arange(8, 18)),
            ],
            3,
            id="unequal-validation-lengths",
        ),
    ],
)
def test_cv_mse_summaries_use_equal_weight_for_every_materialized_split(
    cv: object,
    expected_n_splits: int,
) -> None:
    X, Y = _data(n_samples=18)
    search = PiPLSSearchCV(
        n_components_values=[1],
        predictor_rank_values=[1],
        search_method="optimal",
        cv=cv,
        n_jobs=1,
    ).fit(X, Y)

    split_mse = np.array(
        [
            search.cv_results_[f"split{split_index}_response_standardized_mse"][0]
            for split_index in range(expected_n_splits)
        ],
        dtype=np.float64,
    )
    path = search.component_path_

    assert search.n_splits_ == expected_n_splits
    assert path.n_splits == expected_n_splits
    assert path.cv_mse_mean[0] == pytest.approx(np.mean(split_mse))
    assert path.cv_mse_std[0] == pytest.approx(np.std(split_mse, ddof=0))
    assert search.cv_results_["mean_response_standardized_mse"][0] == pytest.approx(
        np.mean(split_mse)
    )
    assert search.cv_results_["std_response_standardized_mse"][0] == pytest.approx(
        np.std(split_mse, ddof=0)
    )

    if expected_n_splits == 1:
        assert path.cv_mse_std[0] == 0.0
        with pytest.raises(ValueError, match="requires at least two"):
            _ = path.cv_mse_standard_error
    else:
        np.testing.assert_allclose(
            path.cv_mse_standard_error,
            path.cv_mse_std / np.sqrt(expected_n_splits - 1),
        )


def test_predictor_rank_profile_is_sorted_and_consistent_with_cv_results() -> None:
    X, Y = _data()
    search = PiPLSSearchCV(
        n_components_values=[1, 2],
        predictor_rank_values=[1, 2, 3, 4],
        search_method="optimal",
        cv=3,
        n_jobs=1,
    ).fit(X, Y)

    profile = search.predictor_rank_profile(2)

    assert isinstance(profile, PiPLSPredictorRankProfile)
    assert profile.n_components == 2
    assert profile.n_splits == 3
    np.testing.assert_allclose(
        profile.cv_mse_standard_error,
        profile.cv_mse_std / np.sqrt(profile.n_splits - 1),
    )
    assert not profile.cv_mse_standard_error.flags.writeable
    np.testing.assert_array_equal(profile.predictor_rank, np.array([2, 3, 4]))
    assert all(
        not array.flags.writeable
        for array in (
            profile.predictor_rank,
            profile.mean_test_score,
            profile.cv_mse_mean,
            profile.cv_mse_std,
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
        profile.cv_mse_std,
        search.cv_results_["std_response_standardized_mse"][indices],
    )
    assert profile.selection == search.select(n_components=2)


def test_predictor_rank_profile_requires_fitted_evaluated_component_count() -> None:
    search = PiPLSSearchCV()

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
    assert profile.selection.predictor_rank == search.select(
        n_components=2
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


def test_oof_report_uses_existing_model_selection() -> None:
    X, Y = _one_standard_error_data()
    search = PiPLSSearchCV(
        n_components_values=[1, 2, 3],
        predictor_rank_values=[1, 2, 3, 4],
        search_method="optimal",
        cv=4,
        n_jobs=1,
    ).fit(X, Y)
    model = search.refit(X, Y, rule="one_standard_error")

    before = pickle.dumps(search)
    report = search.oof_report(X, Y, selection=model.selection_)

    assert report.selection is model.selection_
    assert report.selection.rule == "one_standard_error"
    assert report.selection.reference_minimum == search.select(
        rule="minimum_cv_mse"
    )
    assert report.selection.one_standard_error_threshold is not None
    assert pickle.dumps(search) == before



def test_oof_report_rejects_non_result_and_incompatible_selection() -> None:
    X, Y = _data()
    search = PiPLSSearchCV(
        n_components_values=[1, 2],
        predictor_rank_values=[1, 2, 3],
        search_method="optimal",
        cv=3,
        n_jobs=1,
    ).fit(X, Y)

    with pytest.raises(TypeError, match="selection must be a PiPLSSelection"):
        search.oof_report(X, Y, selection=object())  # type: ignore[arg-type]

    other = PiPLSSearchCV(
        n_components_values=[1],
        predictor_rank_values=[1],
        search_method="optimal",
        cv=3,
        n_jobs=1,
    ).fit(X, Y)
    with pytest.raises(ValueError, match="not compatible with this fitted search"):
        search.oof_report(
            X,
            Y,
            selection=other.select(n_components=1),
        )


def test_oof_report_requires_fitted_search_and_matching_data_shape() -> None:
    X, Y = _data()
    unfitted = PiPLSSearchCV()
    selection = PiPLSSelection(
        n_components=1,
        predictor_rank=1,
        predictor_rank_policy="optimized",
        mean_test_score=-1.0,
        cv_mse_mean=1.0,
        cv_mse_std=0.1,
        n_splits=3,
    )
    with pytest.raises(NotFittedError):
        unfitted.oof_report(X, Y, selection=selection)

    search = PiPLSSearchCV(
        n_components_values=[1],
        predictor_rank_values=[1],
        search_method="optimal",
        cv=3,
        n_jobs=1,
    ).fit(X, Y)
    compatible = search.select(n_components=1)
    with pytest.raises(ValueError, match=r"oof_report\(\) requires the same number"):
        search.oof_report(X[:-1], Y[:-1], selection=compatible)
    with pytest.raises(
        ValueError,
        match=r"oof_report\(\) requires the same number of response columns",
    ):
        search.oof_report(X, Y[:, :1], selection=compatible)
