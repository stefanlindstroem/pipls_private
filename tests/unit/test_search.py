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
    PiPLSPredictorRankEvidence,
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


def _selection_data() -> tuple[np.ndarray, np.ndarray]:
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


def test_default_adaptive_path_can_achieve_exhaustive_coverage() -> None:
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


def test_search_method_default_and_parameter_surface_use_current_values() -> None:
    search = PiPLSSearchCV()

    assert search.search_method == "adaptive"
    assert search.get_params(deep=False)["search_method"] == "adaptive"
    assert clone(search).search_method == "adaptive"
    search.set_params(search_method="exhaustive")
    assert search.search_method == "exhaustive"
    assert pickle.loads(pickle.dumps(search)).search_method == "exhaustive"
    assert "search_method='exhaustive'" in repr(search)


@pytest.mark.parametrize("retired_value", ["auto", "optimal"])
def test_retired_search_method_values_are_rejected(retired_value: str) -> None:
    X, Y = _data()

    with pytest.raises(
        ValueError,
        match='search_method must be \"adaptive\" or \"exhaustive\"',
    ):
        PiPLSSearchCV(search_method=retired_value).fit(X, Y)  # type: ignore[arg-type]


@pytest.mark.parametrize("predictor_rank_values", ([2], "max"))
def test_one_candidate_rank_policies_reject_exhaustive_search(
    predictor_rank_values: object,
) -> None:
    X, Y = _data()

    with pytest.raises(
        ValueError,
        match='search_method="exhaustive" requires an optimized predictor-rank policy',
    ):
        PiPLSSearchCV(
            predictor_rank_values=predictor_rank_values,  # type: ignore[arg-type]
            search_method="exhaustive",
            cv=3,
        ).fit(X, Y)


@pytest.mark.parametrize("predictor_rank_values", ([2], "max"))
def test_one_candidate_rank_policies_accept_default_adaptive_search(
    predictor_rank_values: object,
) -> None:
    X, Y = _data()
    search = PiPLSSearchCV(
        n_components_values=[1, 2],
        predictor_rank_values=predictor_rank_values,  # type: ignore[arg-type]
        cv=3,
        n_jobs=1,
    ).fit(X, Y)

    assert search.search_method == "adaptive"
    assert search.search_is_exhaustive_


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
        search_method="exhaustive",
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
        search_method="exhaustive",
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
        search_method="exhaustive",
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
        search_method="exhaustive",
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
        search_method="exhaustive",
        cv=3,
    ).fit(X, Y)
    explicit_search = PiPLSSearchCV(
        n_components_values=[1, 2, 3],
        predictor_rank_values=[1, 2, 3],
        max_predictor_rank=3,
        search_method="exhaustive",
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
    X, Y = _selection_data()
    search = PiPLSSearchCV(
        n_components_values=[1, 2, 3],
        predictor_rank_values=[1, 2, 3, 4],
        search_method="exhaustive",
        cv=4,
        n_jobs=1,
    ).fit(X, Y)
    state_before = dict(search.__dict__)

    selected = search.select(n_components=np.int64(2))
    assert selected == search.predictor_rank_profile(2).selection
    assert selected.rule is None
    assert selected.reference_minimum is None
    assert not hasattr(selected, "one_standard_error_threshold")

    best = search.select(rule="best_score")
    best_row = search.select(n_components=best.n_components)
    assert best.rule == "best_score"
    assert best.reference_minimum is None
    assert not hasattr(best, "one_standard_error_threshold")
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
    assert minimum.reference_minimum == search.select(
        n_components=minimum.reference_minimum.n_components
    )
    assert minimum.relative_tolerance == pytest.approx(
        np.sqrt(np.finfo(np.float64).eps)
    )
    assert np.isposinf(minimum.absolute_tolerance)
    assert minimum.cv_mse_threshold is not None
    assert not hasattr(minimum, "one_standard_error_threshold")
    assert not hasattr(minimum, "cv_mse_standard_error")

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
    with pytest.raises(ValueError, match="rule must be"):
        search.select(rule="one_standard_error")  # type: ignore[arg-type]
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

    reference = path._selection_at_index(1)
    assert selected == PiPLSSelection(
        n_components=2,
        predictor_rank=4,
        predictor_rank_policy="optimized",
        mean_test_score=-0.4,
        cv_mse_mean=0.4,
        cv_mse_std=0.08,
        n_splits=5,
        rule="minimum_cv_mse",
        reference_minimum=reference,
        relative_tolerance=np.sqrt(np.finfo(np.float64).eps),
        absolute_tolerance=np.inf,
    )


def test_select_minimum_cv_mse_default_resolves_machine_tolerance() -> None:
    minimum = 0.4
    default = np.sqrt(np.finfo(np.float64).eps)
    path = PiPLSComponentPath(
        n_components=[1, 2, 3],
        predictor_rank=[2, 3, 4],
        predictor_rank_policy="optimized",
        mean_test_score=[-(minimum * (1.0 + 0.5 * default)), -0.5, -minimum],
        cv_mse_mean=[minimum * (1.0 + 0.5 * default), 0.5, minimum],
        cv_mse_std=[0.1, 0.1, 0.1],
        n_splits=5,
    )
    search = _search_with_component_path(path)

    selected = search.select(rule="minimum_cv_mse")

    assert selected.n_components == 1
    assert selected.reference_minimum == path._selection_at_index(2)
    assert selected.relative_tolerance == pytest.approx(default)
    assert np.isposinf(selected.absolute_tolerance)
    assert selected.cv_mse_threshold == pytest.approx(minimum * (1.0 + default))


def test_select_minimum_cv_mse_zero_tolerances_select_exact_minimum() -> None:
    path = PiPLSComponentPath(
        n_components=[1, 2, 3],
        predictor_rank=[2, 3, 4],
        predictor_rank_policy="optimized",
        mean_test_score=[-0.41, -0.40, -0.42],
        cv_mse_mean=[0.41, 0.40, 0.42],
        cv_mse_std=[0.1, 0.1, 0.1],
        n_splits=5,
    )
    search = _search_with_component_path(path)

    selected = search.select(
        rule="minimum_cv_mse",
        relative_tolerance=0.0,
        absolute_tolerance=0.0,
    )

    assert selected.n_components == 2
    assert selected.reference_minimum == path._selection_at_index(1)
    assert selected.cv_mse_threshold == pytest.approx(0.40)


def test_select_minimum_cv_mse_requires_both_tolerances() -> None:
    path = PiPLSComponentPath(
        n_components=[1, 2, 3],
        predictor_rank=[2, 3, 4],
        predictor_rank_policy="optimized",
        mean_test_score=[-1.05, -1.02, -1.00],
        cv_mse_mean=[1.05, 1.02, 1.00],
        cv_mse_std=[0.1, 0.1, 0.1],
        n_splits=5,
    )
    search = _search_with_component_path(path)

    selected = search.select(
        rule="minimum_cv_mse",
        relative_tolerance=0.10,
        absolute_tolerance=0.02,
    )

    assert selected.n_components == 2
    assert selected.cv_mse_threshold == pytest.approx(1.02)
    assert selected.relative_tolerance == pytest.approx(0.10)
    assert selected.absolute_tolerance == pytest.approx(0.02)


def test_select_minimum_cv_mse_includes_exact_threshold_only() -> None:
    minimum = 0.4
    threshold = minimum * 1.10
    path = PiPLSComponentPath(
        n_components=[1, 2, 3],
        predictor_rank=[2, 3, 4],
        predictor_rank_policy="optimized",
        mean_test_score=[-np.nextafter(threshold, np.inf), -threshold, -minimum],
        cv_mse_mean=[np.nextafter(threshold, np.inf), threshold, minimum],
        cv_mse_std=[0.1, 0.1, 0.1],
        n_splits=5,
    )
    search = _search_with_component_path(path)

    selected = search.select(
        rule="minimum_cv_mse",
        relative_tolerance=0.10,
    )

    assert selected.n_components == 2
    assert selected.cv_mse_threshold == pytest.approx(threshold)


def test_select_minimum_cv_mse_zero_minimum_disables_relative_allowance() -> None:
    path = PiPLSComponentPath(
        n_components=[1, 2],
        predictor_rank=[2, 3],
        predictor_rank_policy="optimized",
        mean_test_score=[-1e-12, 0.0],
        cv_mse_mean=[1e-12, 0.0],
        cv_mse_std=[0.0, 0.0],
        n_splits=5,
    )
    search = _search_with_component_path(path)

    selected = search.select(
        rule="minimum_cv_mse",
        relative_tolerance=1.0,
        absolute_tolerance=1.0,
    )

    assert selected.n_components == 2
    assert selected.cv_mse_threshold == 0.0


@pytest.mark.parametrize(
    ("name", "value", "message"),
    [
        ("relative_tolerance", -0.1, "finite nonnegative"),
        ("relative_tolerance", np.inf, "finite nonnegative"),
        ("relative_tolerance", np.nan, "finite nonnegative"),
        ("relative_tolerance", True, "finite nonnegative"),
        ("absolute_tolerance", -0.1, "nonnegative real"),
        ("absolute_tolerance", np.nan, "nonnegative real"),
        ("absolute_tolerance", -np.inf, "nonnegative real"),
        ("absolute_tolerance", False, "nonnegative real"),
    ],
)
def test_select_minimum_cv_mse_rejects_invalid_tolerances(
    name: str,
    value: object,
    message: str,
) -> None:
    path = PiPLSComponentPath(
        n_components=[1],
        predictor_rank=[1],
        predictor_rank_policy="fixed",
        mean_test_score=[-0.4],
        cv_mse_mean=[0.4],
        cv_mse_std=[0.1],
        n_splits=5,
    )
    search = _search_with_component_path(path)
    kwargs = {name: value}

    with pytest.raises(ValueError, match=message):
        search.select(rule="minimum_cv_mse", **kwargs)  # type: ignore[arg-type]


@pytest.mark.parametrize("rule", ["best_score"])
def test_nonminimum_rules_reject_tolerance_arguments(rule: str) -> None:
    path = PiPLSComponentPath(
        n_components=[1, 2],
        predictor_rank=[2, 3],
        predictor_rank_policy="optimized",
        mean_test_score=[-0.5, -0.4],
        cv_mse_mean=[0.5, 0.4],
        cv_mse_std=[0.1, 0.1],
        n_splits=5,
    )
    search = _search_with_component_path(path)

    with pytest.raises(ValueError, match="supported only"):
        search.select(rule=rule, relative_tolerance=0.1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="supported only"):
        search.select(rule=rule, absolute_tolerance=0.1)  # type: ignore[arg-type]


def test_manual_selection_rejects_tolerance_arguments() -> None:
    path = PiPLSComponentPath(
        n_components=[1],
        predictor_rank=[1],
        predictor_rank_policy="fixed",
        mean_test_score=[-0.4],
        cv_mse_mean=[0.4],
        cv_mse_std=[0.1],
        n_splits=5,
    )
    search = _search_with_component_path(path)

    with pytest.raises(ValueError, match="supported only"):
        search.select(n_components=1, relative_tolerance=0.1)
    with pytest.raises(ValueError, match="supported only"):
        search.select(n_components=1, absolute_tolerance=0.1)


def test_minimum_cv_mse_selection_supports_one_validation_split() -> None:
    search = _search_with_component_path(
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

    selected = search.select(rule="minimum_cv_mse")

    assert selected.n_components == 1
    assert selected.cv_mse_std == 0.0
    assert not hasattr(search.component_path_, "cv_mse_standard_error")

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
        search_method="exhaustive",
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
    X, Y = _selection_data()
    search = PiPLSSearchCV(
        n_components_values=[1, 2, 3],
        predictor_rank_values=[1, 2, 3, 4],
        search_method="exhaustive",
        cv=4,
        n_jobs=1,
    ).fit(X, Y)

    expected_by_rule = {
        rule: search.select(rule=rule)
        for rule in ("best_score", "minimum_cv_mse")
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
    X, Y = _selection_data()
    search = PiPLSSearchCV(
        n_components_values=[1, 2, 3],
        predictor_rank_values=[1, 2, 3, 4],
        search_method="exhaustive",
        cv=4,
        n_jobs=1,
    ).fit(X, Y)

    expected = search.select(
        rule="minimum_cv_mse",
        relative_tolerance=0.10,
    )
    model = search.refit(
        X,
        Y,
        rule="minimum_cv_mse",
        relative_tolerance=0.10,
    )
    report = search.oof_report(X, Y, selection=model.selection_)

    assert report.selection == expected
    assert report.selection.rule == "minimum_cv_mse"
    assert report.selection.reference_minimum is not None
    assert report.selection.cv_mse_threshold is not None
    assert model.n_components == expected.n_components
    assert model.predictor_rank == expected.predictor_rank
    assert model.selection_ == expected
    assert model.selection_ == report.selection
    for name in ("selected_params_", "oof_report_"):
        assert not hasattr(search, name)


def test_refit_and_oof_report_preserve_custom_tolerance_provenance() -> None:
    X, Y = _selection_data()
    search = PiPLSSearchCV(
        n_components_values=[1, 2, 3],
        predictor_rank_values=[1, 2, 3, 4],
        search_method="exhaustive",
        cv=4,
        n_jobs=1,
    ).fit(X, Y)

    before = pickle.dumps(search)
    expected = search.select(
        rule="minimum_cv_mse",
        relative_tolerance=0.10,
        absolute_tolerance=0.05,
    )
    model = search.refit(
        X,
        Y,
        rule="minimum_cv_mse",
        relative_tolerance=0.10,
        absolute_tolerance=0.05,
    )
    report = search.oof_report(X, Y, selection=model.selection_)

    assert model.selection_ == expected
    assert report.selection == expected
    assert report.selection.relative_tolerance == pytest.approx(0.10)
    assert report.selection.absolute_tolerance == pytest.approx(0.05)
    restored = pickle.loads(pickle.dumps(model))
    assert restored.selection_ == expected
    assert pickle.dumps(search) == before


def test_oof_report_rejects_changed_tolerance_provenance() -> None:
    X, Y = _selection_data()
    search = PiPLSSearchCV(
        n_components_values=[1, 2, 3],
        predictor_rank_values=[1, 2, 3, 4],
        search_method="exhaustive",
        cv=4,
        n_jobs=1,
    ).fit(X, Y)
    selected = search.select(
        rule="minimum_cv_mse",
        relative_tolerance=0.10,
    )
    false_reference = PiPLSSelection(
        n_components=selected.n_components,
        predictor_rank=selected.predictor_rank,
        predictor_rank_policy=selected.predictor_rank_policy,
        mean_test_score=selected.mean_test_score,
        cv_mse_mean=selected.cv_mse_mean,
        cv_mse_std=selected.cv_mse_std,
        n_splits=selected.n_splits,
    )
    incompatible = PiPLSSelection(
        n_components=selected.n_components,
        predictor_rank=selected.predictor_rank,
        predictor_rank_policy=selected.predictor_rank_policy,
        mean_test_score=selected.mean_test_score,
        cv_mse_mean=selected.cv_mse_mean,
        cv_mse_std=selected.cv_mse_std,
        n_splits=selected.n_splits,
        rule="minimum_cv_mse",
        reference_minimum=false_reference,
        relative_tolerance=0.10,
        absolute_tolerance=np.inf,
    )

    with pytest.raises(ValueError, match="not compatible"):
        search.oof_report(X, Y, selection=incompatible)


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
    with pytest.raises(ValueError, match="rule must be"):
        search.refit(X, Y, rule="one_standard_error")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="was not evaluated"):
        search.refit(X, Y, n_components=3)


def test_post_fit_refit_requires_a_fitted_search() -> None:
    X, Y = _data()
    search = PiPLSSearchCV()

    with pytest.raises(NotFittedError):
        search.refit(X, Y, rule="best_score")


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
    X, Y = _selection_data()
    search = PiPLSSearchCV(
        n_components_values=[1, 2, 3],
        predictor_rank_values=[1, 2, 3, 4],
        search_method="exhaustive",
        cv=4,
        n_jobs=1,
    ).fit(X, Y)

    model = search.refit(
        X,
        Y,
        rule="minimum_cv_mse",
        relative_tolerance=0.10,
    )
    restored = pickle.loads(pickle.dumps(model))
    cloned = clone(model)

    assert restored.selection_ == model.selection_
    assert restored.selection_.reference_minimum == search.select(
        rule="minimum_cv_mse"
    ).reference_minimum
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


def test_adaptive_path_skips_candidates_with_constant_scorer() -> None:
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
        search_method="adaptive",
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
        search_method="exhaustive",
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
        ("search_method", "unsupported", "search_method"),
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
        search_method="exhaustive",
        cv=3,
        n_jobs=1,
    ).fit(X, Y)

    path = search.component_path_
    assert isinstance(path, PiPLSComponentPath)
    np.testing.assert_array_equal(path.n_components, np.array([1, 2]))
    assert path.n_splits == 3
    assert path.predictor_rank_policy == "optimized"
    assert path.predictor_rank_evidence is not None
    assert len(path.predictor_rank_evidence) == path.n_components.size

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
    assert not hasattr(path, "cv_mse_standard_error")


def test_predictor_rank_profile_is_sorted_and_consistent_with_cv_results() -> None:
    X, Y = _data()
    search = PiPLSSearchCV(
        n_components_values=[1, 2],
        predictor_rank_values=[1, 2, 3, 4],
        search_method="exhaustive",
        cv=3,
        n_jobs=1,
    ).fit(X, Y)

    profile = search.predictor_rank_profile(2)

    assert isinstance(profile, PiPLSPredictorRankProfile)
    assert profile.n_components == 2
    assert profile.n_splits == 3
    assert not hasattr(profile, "cv_mse_standard_error")
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
    if expected_policy == "optimized":
        assert search.component_path_.predictor_rank_evidence is not None
    else:
        assert search.component_path_.predictor_rank_evidence is None
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
    X, Y = _selection_data()
    search = PiPLSSearchCV(
        n_components_values=[1, 2, 3],
        predictor_rank_values=[1, 2, 3, 4],
        search_method="exhaustive",
        cv=4,
        n_jobs=1,
    ).fit(X, Y)
    model = search.refit(
        X,
        Y,
        rule="minimum_cv_mse",
        relative_tolerance=0.10,
    )

    before = pickle.dumps(search)
    report = search.oof_report(X, Y, selection=model.selection_)

    assert report.selection is model.selection_
    assert report.selection.rule == "minimum_cv_mse"
    assert report.selection.reference_minimum is not None
    assert report.selection.cv_mse_threshold is not None
    assert pickle.dumps(search) == before



def test_oof_report_rejects_non_result_and_incompatible_selection() -> None:
    X, Y = _data()
    search = PiPLSSearchCV(
        n_components_values=[1, 2],
        predictor_rank_values=[1, 2, 3],
        search_method="exhaustive",
        cv=3,
        n_jobs=1,
    ).fit(X, Y)

    with pytest.raises(TypeError, match="selection must be a PiPLSSelection"):
        search.oof_report(X, Y, selection=object())  # type: ignore[arg-type]

    other = PiPLSSearchCV(
        n_components_values=[1],
        predictor_rank_values=[1],
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


def test_predictor_rank_relative_tolerance_conditions_the_component_path() -> None:
    X, Y = _data()
    score_by_rank = {1: 0.91, 2: 1.0, 3: 0.95}

    def scorer(estimator: object, X_validation: object, y_validation: object) -> float:
        del X_validation, y_validation
        return score_by_rank[int(estimator.predictor_rank)]

    search = PiPLSSearchCV(
        n_components_values=[1],
        predictor_rank_values=[1, 2, 3],
        max_predictor_rank=3,
        search_method="exhaustive",
        predictor_rank_relative_tolerance=0.10,
        scoring=scorer,
        cv=3,
        n_jobs=1,
    ).fit(X, Y)

    selected = search.select(n_components=1)
    profile = search.predictor_rank_profile(1)
    evidence = selected.predictor_rank_evidence

    assert isinstance(evidence, PiPLSPredictorRankEvidence)
    assert evidence.reference_predictor_rank == 2
    assert evidence.reference_mean_test_score == pytest.approx(1.0)
    assert evidence.relative_tolerance == pytest.approx(0.10)
    assert np.isposinf(evidence.absolute_tolerance)
    assert evidence.score_threshold == pytest.approx(0.90)
    assert selected.predictor_rank == 1
    assert selected.mean_test_score == pytest.approx(0.91)
    assert profile.reference_selection.predictor_rank == 2
    assert profile.reference_selection.predictor_rank_evidence is None
    assert profile.selection == selected
    assert search.component_path_.predictor_rank_evidence == (evidence,)

    model = search.refit(X, Y, n_components=1)
    report = search.oof_report(X, Y, selection=model.selection_)
    assert model.selection_ == selected
    assert report.selection == selected
    assert report.selection.predictor_rank_evidence is evidence


def test_predictor_rank_absolute_tolerance_can_control_qualification() -> None:
    X, Y = _data()
    score_by_rank = {1: 0.91, 2: 1.0, 3: 0.96}

    def scorer(estimator: object, X_validation: object, y_validation: object) -> float:
        del X_validation, y_validation
        return score_by_rank[int(estimator.predictor_rank)]

    search = PiPLSSearchCV(
        n_components_values=[1],
        predictor_rank_values=[1, 2, 3],
        max_predictor_rank=3,
        search_method="exhaustive",
        predictor_rank_relative_tolerance=0.10,
        predictor_rank_absolute_tolerance=0.05,
        scoring=scorer,
        cv=3,
        n_jobs=1,
    ).fit(X, Y)

    selected = search.select(n_components=1)
    assert selected.predictor_rank == 2
    assert selected.predictor_rank_evidence is not None
    assert selected.predictor_rank_evidence.score_threshold == pytest.approx(0.95)


def test_best_score_selects_from_the_conditioned_component_path() -> None:
    X, Y = _data()
    score_by_pair = {
        (1, 1): 0.91,
        (1, 2): 1.00,
        (1, 3): 0.95,
        (2, 2): 0.95,
        (2, 3): 0.94,
    }

    def scorer(estimator: object, X_validation: object, y_validation: object) -> float:
        del X_validation, y_validation
        key = (int(estimator.n_components), int(estimator.predictor_rank))
        return score_by_pair[key]

    search = PiPLSSearchCV(
        n_components_values=[1, 2],
        predictor_rank_values=[1, 2, 3],
        max_predictor_rank=3,
        search_method="exhaustive",
        predictor_rank_relative_tolerance=0.10,
        scoring=scorer,
        cv=3,
        n_jobs=1,
    ).fit(X, Y)

    global_index = int(np.argmax(search.cv_results_["mean_test_score"]))
    assert tuple(
        int(search.cv_results_[name][global_index])
        for name in ("n_components", "predictor_rank")
    ) == (1, 2)
    np.testing.assert_array_equal(search.component_path_.predictor_rank, [1, 2])
    np.testing.assert_allclose(search.component_path_.mean_test_score, [0.91, 0.95])

    best = search.select(rule="best_score")
    assert (best.n_components, best.predictor_rank) == (2, 2)
    assert best.mean_test_score == pytest.approx(0.95)


def test_predictor_rank_tolerance_does_not_change_adaptive_candidate_coverage() -> None:
    X, Y = _data(n_samples=80)
    common = {
        "n_components_values": [1],
        "predictor_rank_values": list(range(1, 9)),
        "max_predictor_rank": 8,
        "search_method": "adaptive",
        "cv": 3,
        "n_jobs": 1,
    }
    exact = PiPLSSearchCV(
        **common,
        predictor_rank_relative_tolerance=0.0,
        predictor_rank_absolute_tolerance=0.0,
    ).fit(X, Y)
    tolerant = PiPLSSearchCV(
        **common,
        predictor_rank_relative_tolerance=0.50,
        predictor_rank_absolute_tolerance=np.inf,
    ).fit(X, Y)

    for name in ("n_components", "predictor_rank"):
        np.testing.assert_array_equal(exact.cv_results_[name], tolerant.cv_results_[name])
    for name in exact.cv_results_:
        if name.startswith("split") and name.endswith("_test_score"):
            np.testing.assert_allclose(exact.cv_results_[name], tolerant.cv_results_[name])
    assert exact.search_is_exhaustive_ == tolerant.search_is_exhaustive_


@pytest.mark.parametrize("predictor_rank_values", ([3], "max"))
@pytest.mark.parametrize(
    "tolerance_kwargs",
    (
        {"predictor_rank_relative_tolerance": 0.10},
        {"predictor_rank_absolute_tolerance": 0.10},
    ),
)
def test_nondefault_predictor_rank_tolerances_require_optimized_policy(
    predictor_rank_values: object,
    tolerance_kwargs: dict[str, float],
) -> None:
    X, Y = _data()
    with pytest.raises(ValueError, match="optimized predictor-rank policy"):
        PiPLSSearchCV(
            n_components_values=[1, 2],
            predictor_rank_values=predictor_rank_values,
            cv=3,
            **tolerance_kwargs,
        ).fit(X, Y)


@pytest.mark.parametrize(
    ("keyword", "value", "message"),
    [
        ("predictor_rank_relative_tolerance", -0.1, "finite nonnegative"),
        ("predictor_rank_relative_tolerance", np.inf, "finite nonnegative"),
        ("predictor_rank_relative_tolerance", np.nan, "finite nonnegative"),
        ("predictor_rank_relative_tolerance", True, "finite nonnegative"),
        ("predictor_rank_absolute_tolerance", -0.1, "nonnegative real"),
        ("predictor_rank_absolute_tolerance", np.nan, "nonnegative real"),
        ("predictor_rank_absolute_tolerance", -np.inf, "nonnegative real"),
        ("predictor_rank_absolute_tolerance", False, "nonnegative real"),
    ],
)
def test_invalid_predictor_rank_tolerances_are_rejected(
    keyword: str,
    value: object,
    message: str,
) -> None:
    X, Y = _data()
    with pytest.raises(ValueError, match=message):
        PiPLSSearchCV(**{keyword: value}).fit(X, Y)


def test_predictor_rank_tolerance_parameters_clone_repr_and_pickle() -> None:
    search = PiPLSSearchCV(
        predictor_rank_relative_tolerance=0.10,
        predictor_rank_absolute_tolerance=0.25,
    )
    cloned = clone(search)
    restored = pickle.loads(pickle.dumps(search))

    for result in (cloned, restored):
        assert result.predictor_rank_relative_tolerance == pytest.approx(0.10)
        assert result.predictor_rank_absolute_tolerance == pytest.approx(0.25)
    assert "predictor_rank_relative_tolerance=0.1" in repr(search)
    assert "predictor_rank_absolute_tolerance=0.25" in repr(search)
