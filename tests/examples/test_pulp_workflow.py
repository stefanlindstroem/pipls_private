from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from sklearn.model_selection import KFold

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

    cv = KFold(n_splits=5, shuffle=True, random_state=0)
    search = PiPLSSearchCV(cv=cv).fit(X, Y)
    model = search.refit(
        X,
        Y,
        n_components=3,
    )
    selected = model.selection_
    component_path = search.component_path_
    rank_profile = search.predictor_rank_profile(selected.n_components)
    report = search.oof_report(
        X,
        Y,
        selection=selected,
    )
    oof_predictions = report.oof_predictions
    factors = pipls_display_factors(
        model.decomposition_,
        response_index=data.target_names.index("TI"),
        response_sign="positive",
    )
    structure = latent_structure(model)
    diagnostics = prediction_diagnostics(
        Y,
        oof_predictions,
        prediction_kind="selection-conditioned OOF predictions",
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
        diagnostics=diagnostics,
    )


def test_pulp_path_selects_the_documented_fixed_pair(pulp_result: SimpleNamespace) -> None:
    result = pulp_result

    assert isinstance(result.search, PiPLSSearchCV)
    assert result.search.estimator is None
    assert callable(result.search.refit)
    assert isinstance(result.component_path, PiPLSComponentPath)
    assert result.selected.n_components == 3
    assert result.selected.predictor_rank == 9
    assert isinstance(result.model, PiPLSRegression)
    assert result.model.n_components == 3
    assert result.model.predictor_rank_ == 9


def test_pulp_rank_profile_exposes_the_interior_selection(
    pulp_result: SimpleNamespace,
) -> None:
    result = pulp_result

    profile = result.rank_profile
    assert isinstance(profile, PiPLSPredictorRankProfile)
    np.testing.assert_array_equal(profile.predictor_rank, np.arange(3, 11))
    assert profile.selection == result.selected
    assert profile.selection.predictor_rank == 9
    selected_index = int(np.flatnonzero(profile.predictor_rank == 9)[0])
    upper_index = int(np.flatnonzero(profile.predictor_rank == 10)[0])
    assert profile.cv_mse_mean[selected_index] < profile.cv_mse_mean[upper_index]
    assert (
        profile.cv_mse_mean[upper_index] - profile.cv_mse_mean[selected_index]
        < min(
            profile.cv_mse_fold_sd[selected_index],
            profile.cv_mse_fold_sd[upper_index],
        )
    )


def test_pulp_oof_and_inspection_results_are_aligned(pulp_result: SimpleNamespace) -> None:
    result = pulp_result

    assert result.X.shape == (46, 14)
    assert result.Y.shape == (46, 8)
    assert result.report.selection == result.selected
    assert result.report.has_complete_oof_coverage
    assert result.oof_predictions.shape == result.Y.shape
    assert np.all(np.isfinite(result.oof_predictions))
    assert result.diagnostics.prediction_kind == "selection-conditioned OOF predictions"
    ti_response_index = result.data.target_names.index("TI")
    assert np.all(result.factors.response_directions[ti_response_index] > 0.0)
    assert np.all(result.factors.weighted_response_directions[ti_response_index] > 0.0)
    assert result.diagnostics.observed.shape == result.Y.shape
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
