from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from sklearn.model_selection import RepeatedKFold

from pipls import PiPLSRegression, PiPLSSearchCV
from pipls.component_path import PiPLSComponentPath, PiPLSPredictorRankProfile
from pipls.datasets import load_pulp
from pipls.inspection import (
    latent_structure,
    pipls_display_factors,
    prediction_diagnostics,
)


@pytest.fixture(scope="module")
def pulp_result() -> SimpleNamespace:
    data = load_pulp()
    X, Y = data.X, data.Y

    cv = RepeatedKFold(n_splits=5, n_repeats=10, random_state=0)
    search = PiPLSSearchCV(cv=cv).fit(X, Y)
    component_path = search.component_path_
    selected = search.select(n_components=3)
    rank_profile = search.predictor_rank_profile(selected.n_components)
    report = search.oof_report(
        X,
        Y,
        selection=selected,
    )
    oof_predictions = report.oof_predictions
    model = search.refit(X, Y, selection=selected)
    factors = pipls_display_factors(
        model.decomposition_,
        response_index=data.target_names.index("TI"),
        response_sign="positive",
    )
    structure = latent_structure(model)
    oof_diagnostics = prediction_diagnostics(
        Y,
        oof_predictions,
        prediction_kind="selection-conditioned OOF predictions",
    )
    fitted_predictions = model.predict(X)
    fitted_diagnostics = prediction_diagnostics(
        Y,
        fitted_predictions,
        prediction_kind="fitted values",
    )
    return SimpleNamespace(
        data=data,
        X=X,
        Y=Y,
        search=search,
        component_path=component_path,
        selected=selected,
        rank_profile=rank_profile,
        model=model,
        report=report,
        oof_predictions=oof_predictions,
        factors=factors,
        structure=structure,
        oof_diagnostics=oof_diagnostics,
        fitted_predictions=fitted_predictions,
        fitted_diagnostics=fitted_diagnostics,
    )


def test_pulp_path_selects_the_documented_fixed_pair(pulp_result: SimpleNamespace) -> None:
    result = pulp_result

    assert isinstance(result.search, PiPLSSearchCV)
    assert result.search.estimator is None
    assert callable(result.search.refit)
    assert result.search.n_splits_ == 50
    assert isinstance(result.component_path, PiPLSComponentPath)
    assert result.selected.n_components == 3
    assert result.selected.predictor_rank == 9
    assert isinstance(result.model, PiPLSRegression)
    assert result.model.n_components == 3
    assert result.model.predictor_rank == 9
    assert result.model.selection_ is result.selected
    assert result.report.selection is result.selected


def test_pulp_rank_profile_exposes_the_interior_selection(
    pulp_result: SimpleNamespace,
) -> None:
    result = pulp_result

    profile = result.rank_profile
    assert isinstance(profile, PiPLSPredictorRankProfile)
    np.testing.assert_array_equal(profile.predictor_rank, np.arange(3, 15))
    assert profile.selection == result.selected
    assert profile.selection.predictor_rank == 9
    selected_index = int(np.flatnonzero(profile.predictor_rank == 9)[0])
    upper_index = int(np.flatnonzero(profile.predictor_rank == 10)[0])
    assert profile.cv_mse_mean[selected_index] < profile.cv_mse_mean[upper_index]
    assert profile.cv_mse_mean[selected_index] == pytest.approx(0.258427287979069)
    assert profile.cv_mse_mean[upper_index] == pytest.approx(0.2742883932741335)
    assert profile.cv_mse_std[selected_index] == pytest.approx(0.0965526957919972)
    assert profile.cv_mse_std[upper_index] == pytest.approx(0.09984580272725549)
    assert (
        profile.cv_mse_mean[upper_index] - profile.cv_mse_mean[selected_index]
        < min(
            profile.cv_mse_std[selected_index],
            profile.cv_mse_std[upper_index],
        )
    )


def test_pulp_oof_and_inspection_results_are_aligned(pulp_result: SimpleNamespace) -> None:
    result = pulp_result

    assert result.X.shape == (46, 14)
    assert result.Y.shape == (46, 8)
    assert result.report.selection == result.selected
    assert result.report.has_complete_oof_coverage
    np.testing.assert_array_equal(
        result.report.oof_prediction_counts,
        np.full(result.X.shape[0], 10, dtype=np.int64),
    )
    assert result.oof_predictions.shape == result.Y.shape
    assert np.all(np.isfinite(result.oof_predictions))
    assert result.oof_diagnostics.prediction_kind == "selection-conditioned OOF predictions"
    ti_response_index = result.data.target_names.index("TI")
    assert np.all(result.factors.response_directions[ti_response_index] > 0.0)
    assert np.all(result.factors.weighted_response_directions[ti_response_index] > 0.0)
    assert result.oof_diagnostics.observed.shape == result.Y.shape
    assert result.factors.n_components == result.selected.n_components
    assert result.structure.x_scores.shape[0] == len(result.X)
    assert result.structure.x_loadings.shape[0] == result.X.shape[1]
    assert result.structure.y_loadings.shape[0] == result.Y.shape[1]
    assert result.structure.n_components == result.selected.n_components


def test_pulp_selection_requires_an_evaluated_component_count(
    pulp_result: SimpleNamespace,
) -> None:
    with pytest.raises(ValueError, match="was not evaluated"):
        pulp_result.search.select(n_components=99)


def test_pulp_final_fit_diagnostics_are_response_wise_and_descriptive(
    pulp_result: SimpleNamespace,
) -> None:
    result = pulp_result

    diagnostics = result.fitted_diagnostics
    assert diagnostics.prediction_kind == "fitted values"
    np.testing.assert_allclose(diagnostics.predicted, result.fitted_predictions)
    assert diagnostics.predicted.shape == result.Y.shape
    assert diagnostics.response_r2.shape == (result.Y.shape[1],)
    assert np.all(np.isfinite(diagnostics.response_r2))

    residual = result.Y - result.fitted_predictions
    residual_sum_squares = np.sum(residual * residual, axis=0)
    centered = result.Y - np.mean(result.Y, axis=0)
    total_sum_squares = np.sum(centered * centered, axis=0)
    expected_r2 = 1.0 - residual_sum_squares / total_sum_squares
    np.testing.assert_allclose(diagnostics.response_r2, expected_r2)

    pooled_residuals = diagnostics.residual_standardized.ravel()
    assert np.all(np.isfinite(pooled_residuals))
    assert float(np.std(pooled_residuals, ddof=1)) > 0.0
