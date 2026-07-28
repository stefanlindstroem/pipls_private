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

from pipls import PiPLSRegression, PiPLSSearchCV


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
    search = PiPLSSearchCV(
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

    report = search.validation_report_
    assert report.oof_predictions is not None
    assert report.oof_prediction_counts is not None
    np.testing.assert_allclose(report.oof_predictions, expected)
    np.testing.assert_array_equal(report.oof_prediction_counts, np.ones(X.shape[0]))
    assert report.n_components == search.best_n_components_
    assert report.predictor_rank == search.best_predictor_rank_
    assert report.selection_conditioned
    assert report.is_leave_one_out
    assert report.complete_oof_coverage
    assert report.mean_response_standardized_mse == pytest.approx(
        search.cv_results_["mean_response_standardized_mse"][search.best_index_]
    )
    assert report.pooled_oof_r2 == pytest.approx(
        r2_score(Y, expected, multioutput="uniform_average")
    )




def test_repeated_kfold_averages_predictions_and_records_counts() -> None:
    X, Y = _data()
    search = PiPLSSearchCV(
        estimator=_fixed(),
        n_components_values=[1],
        predictor_rank_values=[2],
        max_predictor_rank=2,
        cv=RepeatedKFold(n_splits=3, n_repeats=2, random_state=7),
        return_oof_predictions=True,
        n_jobs=1,
    ).fit(X, Y)

    report = search.validation_report_
    assert report.oof_prediction_counts is not None
    assert report.oof_predictions is not None
    np.testing.assert_array_equal(
        report.oof_prediction_counts,
        np.full(X.shape[0], 2, dtype=np.intp),
    )
    assert report.complete_oof_coverage
    assert np.all(np.isfinite(report.oof_predictions))


def test_predefined_and_temporal_splits_mark_uncovered_rows() -> None:
    X, Y = _data()
    predefined = PiPLSSearchCV(
        estimator=_fixed(),
        n_components_values=[1],
        predictor_rank_values=[2],
        max_predictor_rank=2,
        cv=PredefinedSplit(np.array([-1] * 9 + [0] * 9)),
        return_oof_predictions=True,
    ).fit(X, Y)
    temporal = PiPLSSearchCV(
        estimator=_fixed(),
        n_components_values=[1],
        predictor_rank_values=[2],
        max_predictor_rank=2,
        cv=TimeSeriesSplit(n_splits=3),
        return_oof_predictions=True,
    ).fit(X, Y)

    predefined_report = predefined.validation_report_
    temporal_report = temporal.validation_report_
    assert predefined_report.oof_prediction_counts is not None
    assert predefined_report.oof_predictions is not None
    assert temporal_report.oof_prediction_counts is not None
    assert temporal_report.oof_predictions is not None
    np.testing.assert_array_equal(predefined_report.oof_prediction_counts[:9], 0)
    assert np.isnan(predefined_report.oof_predictions[:9]).all()
    assert not predefined_report.complete_oof_coverage
    assert np.any(temporal_report.oof_prediction_counts == 0)
    uncovered = temporal_report.oof_prediction_counts == 0
    assert np.isnan(temporal_report.oof_predictions[uncovered]).all()
    assert not temporal_report.complete_oof_coverage


def test_grouped_splitters_are_supported_by_path_interface() -> None:
    X, Y = _data(24)
    groups = np.repeat(np.arange(8), 3)
    path = PiPLSSearchCV(
        estimator=_fixed(),
        n_components_values=[1],
        predictor_rank_values=[2],
        max_predictor_rank=2,
        cv=GroupKFold(n_splits=4),
    ).fit(X, Y, groups=groups)

    assert path.n_splits_ == 4


def test_groups_participate_in_path_metadata_routing() -> None:
    X, Y = _data(30)
    groups = np.repeat(np.arange(10), 3)
    estimator = PiPLSSearchCV(
        estimator=_fixed(),
        n_components_values=[1],
        predictor_rank_values=[2],
        max_predictor_rank=2,
        cv=GroupKFold(n_splits=3),
        refit=True,
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


@pytest.mark.parametrize("scoring", [None, "r2"])
def test_singleton_validation_rejects_foldwise_r2(scoring: object) -> None:
    X, Y = _data(10)
    estimator = PiPLSSearchCV(
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
    search = PiPLSSearchCV(
        estimator=_fixed(),
        n_components_values=[1],
        predictor_rank_values=[2],
        max_predictor_rank=2,
        cv=3,
        return_oof_predictions=True,
    ).fit(X, y)

    report = search.validation_report_
    assert report.oof_predictions is not None
    assert report.oof_prediction_counts is not None
    assert report.oof_predictions.shape == (X.shape[0],)
    assert not report.oof_predictions.flags.writeable
    assert not report.oof_prediction_counts.flags.writeable
    with pytest.raises(ValueError, match="read-only"):
        report.oof_predictions[0] = 0.0


def test_return_oof_predictions_requires_boolean() -> None:
    X, Y = _data()
    with pytest.raises(ValueError, match="return_oof_predictions must be boolean"):
        PiPLSSearchCV(return_oof_predictions=1).fit(X, Y)  # type: ignore[arg-type]
