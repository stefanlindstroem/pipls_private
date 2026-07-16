from __future__ import annotations

import numpy as np
import pytest
from sklearn import config_context
from sklearn.base import clone
from sklearn.metrics import r2_score
from sklearn.model_selection import (
    GroupKFold,
    KFold,
    LeaveOneOut,
    PredefinedSplit,
    RepeatedKFold,
    TimeSeriesSplit,
    cross_validate,
)

from pipls import PiPLSPathCV, PiPLSRegression, PiPLSValidationReport


def _data(n_samples: int = 18) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(20260719)
    X = rng.normal(size=(n_samples, 6))
    Y = X @ rng.normal(size=(6, 2)) + 0.05 * rng.normal(size=(n_samples, 2))
    return X, Y


def _fixed() -> PiPLSRegression:
    return PiPLSRegression(
        n_components=1,
        predictor_rank=2,
        scale=False,
        svd_solver="full",
        random_state=None,
    )


def test_path_leave_one_out_predictions_are_ordered_and_selection_conditioned() -> None:
    X, Y = _data(12)
    splitter = LeaveOneOut()
    search = PiPLSPathCV(
        estimator=_fixed(),
        n_components_values=[1],
        predictor_rank_values=[2],
        max_predictor_rank=2,
        search_method="optimal",
        cv=splitter,
        return_oof_predictions=True,
        n_jobs=1,
    ).fit(X, Y)

    expected = np.empty_like(Y)
    for train, validation in splitter.split(X, Y):
        expected[validation] = clone(_fixed()).fit(X[train], Y[train]).predict(X[validation])

    np.testing.assert_allclose(search.oof_predictions_, expected)
    np.testing.assert_array_equal(search.oof_prediction_counts_, np.ones(X.shape[0]))
    assert search.oof_params_ == search.best_params_
    assert search.validation_report_.selection_conditioned
    assert search.validation_report_.is_leave_one_out
    assert search.validation_report_.complete_oof_coverage
    assert search.validation_report_.mean_response_standardized_mse == pytest.approx(
        search.best_response_standardized_mse_
    )
    assert search.pooled_oof_r2_ == pytest.approx(
        r2_score(Y, expected, multioutput="uniform_average")
    )


def test_regression_oof_report_distinguishes_selected_and_fixed_parameters() -> None:
    X, Y = _data(15)
    selected = PiPLSRegression(
        n_components=1,
        predictor_rank="optimal",
        samples_per_predictor_rank=5,
        cv=3,
        return_oof_predictions=True,
        svd_solver="full",
        random_state=None,
    ).fit(X, Y)
    fixed = _fixed().set_params(cv=3, return_oof_predictions=True).fit(X, Y)

    assert isinstance(selected.validation_report_, PiPLSValidationReport)
    assert selected.validation_report_.estimate_kind == "selection-conditioned"
    assert selected.validation_report_.predictor_rank == selected.predictor_rank_
    assert fixed.validation_report_.estimate_kind == "fixed-parameter"
    assert fixed.validation_report_.predictor_rank == 2
    assert fixed.oof_params_ == {"n_components": 1, "predictor_rank": 2}


def test_repeated_kfold_averages_predictions_and_records_counts() -> None:
    X, Y = _data()
    search = PiPLSPathCV(
        estimator=_fixed(),
        n_components_values=[1],
        predictor_rank_values=[2],
        max_predictor_rank=2,
        cv=RepeatedKFold(n_splits=3, n_repeats=2, random_state=7),
        return_oof_predictions=True,
        n_jobs=1,
    ).fit(X, Y)

    np.testing.assert_array_equal(
        search.oof_prediction_counts_,
        np.full(X.shape[0], 2, dtype=np.intp),
    )
    assert search.validation_report_.complete_oof_coverage
    assert np.all(np.isfinite(search.oof_predictions_))


def test_predefined_and_temporal_splits_mark_uncovered_rows() -> None:
    X, Y = _data()
    predefined = PiPLSPathCV(
        estimator=_fixed(),
        n_components_values=[1],
        predictor_rank_values=[2],
        max_predictor_rank=2,
        cv=PredefinedSplit(np.array([-1] * 9 + [0] * 9)),
        return_oof_predictions=True,
    ).fit(X, Y)
    temporal = PiPLSPathCV(
        estimator=_fixed(),
        n_components_values=[1],
        predictor_rank_values=[2],
        max_predictor_rank=2,
        cv=TimeSeriesSplit(n_splits=3),
        return_oof_predictions=True,
    ).fit(X, Y)

    np.testing.assert_array_equal(predefined.oof_prediction_counts_[:9], 0)
    assert np.isnan(predefined.oof_predictions_[:9]).all()
    assert not predefined.validation_report_.complete_oof_coverage
    assert np.any(temporal.oof_prediction_counts_ == 0)
    assert np.isnan(temporal.oof_predictions_[temporal.oof_prediction_counts_ == 0]).all()
    assert not temporal.validation_report_.complete_oof_coverage


def test_grouped_splitters_are_supported_by_both_public_interfaces() -> None:
    X, Y = _data(24)
    groups = np.repeat(np.arange(8), 3)
    regression = PiPLSRegression(
        n_components=1,
        predictor_rank="optimal",
        samples_per_predictor_rank=5,
        cv=GroupKFold(n_splits=4),
        svd_solver="full",
        random_state=None,
    ).fit(X, Y, groups=groups)
    path = PiPLSPathCV(
        estimator=_fixed(),
        n_components_values=[1],
        predictor_rank_values=[2],
        max_predictor_rank=2,
        cv=GroupKFold(n_splits=4),
    ).fit(X, Y, groups=groups)

    assert regression.n_splits_ == 4
    assert path.n_splits_ == 4


@pytest.mark.parametrize("estimator_kind", ["regression", "path"])
def test_groups_participate_in_sklearn_metadata_routing(
    estimator_kind: str,
) -> None:
    X, Y = _data(30)
    groups = np.repeat(np.arange(10), 3)
    if estimator_kind == "regression":
        estimator = PiPLSRegression(
            n_components=1,
            predictor_rank="optimal",
            samples_per_predictor_rank=5,
            cv=GroupKFold(n_splits=3),
            svd_solver="full",
            random_state=None,
        )
    else:
        estimator = PiPLSPathCV(
            estimator=_fixed(),
            n_components_values=[1],
            predictor_rank_values=[2],
            max_predictor_rank=2,
            cv=GroupKFold(n_splits=3),
        )

    with config_context(enable_metadata_routing=True):
        routed = estimator.set_fit_request(groups=True)
        result = cross_validate(
            routed,
            X,
            Y,
            cv=KFold(n_splits=3),
            params={"groups": groups},
        )
    assert result["test_score"].shape == (3,)


@pytest.mark.parametrize("estimator_kind", ["regression", "path"])
@pytest.mark.parametrize("scoring", [None, "r2"])
def test_singleton_validation_rejects_foldwise_r2(
    estimator_kind: str,
    scoring: object,
) -> None:
    X, Y = _data(10)
    if estimator_kind == "regression":
        estimator = PiPLSRegression(
            n_components=1,
            predictor_rank="optimal",
            samples_per_predictor_rank=5,
            cv=LeaveOneOut(),
            scoring=scoring,
            svd_solver="full",
            random_state=None,
        )
    else:
        estimator = PiPLSPathCV(
            estimator=_fixed(),
            n_components_values=[1],
            predictor_rank_values=[2],
            max_predictor_rank=2,
            cv=LeaveOneOut(),
            scoring=scoring,
        )

    with pytest.raises(ValueError, match="R2 scoring is undefined"):
        estimator.fit(X, Y)


def test_oof_arrays_are_read_only_and_one_dimensional_targets_stay_one_dimensional() -> None:
    X, Y = _data()
    y = Y[:, 0]
    model = _fixed().set_params(cv=3, return_oof_predictions=True).fit(X, y)

    assert model.oof_predictions_.shape == (X.shape[0],)
    assert not model.oof_predictions_.flags.writeable
    assert not model.oof_prediction_counts_.flags.writeable
    with pytest.raises(ValueError, match="read-only"):
        model.oof_predictions_[0] = 0.0


@pytest.mark.parametrize("estimator", [PiPLSRegression(), PiPLSPathCV()])
def test_return_oof_predictions_requires_boolean(estimator: object) -> None:
    X, Y = _data()
    with pytest.raises(ValueError, match="return_oof_predictions must be boolean"):
        estimator.set_params(return_oof_predictions=1).fit(X, Y)
