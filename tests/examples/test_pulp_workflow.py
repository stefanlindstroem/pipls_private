from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import KFold, cross_val_predict

from pipls import PiPLSComponentPath, PiPLSPathCV, PiPLSRegression
from pipls.inspection import (
    latent_structure,
    pipls_display_factors,
    prediction_diagnostics,
)


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def pulp_result() -> SimpleNamespace:
    data_dir = _repository_root() / "datasets" / "pulp"
    X = pd.read_csv(data_dir / "X.csv")
    Y = pd.read_csv(data_dir / "Y.csv")

    path_search = PiPLSPathCV(refit=False).fit(X, Y)
    component_path = path_search.component_path_
    selected = component_path.for_n_components(3)

    cv_results = path_search.cv_results_
    rank_rows = np.asarray(cv_results["n_components"]) == selected.n_components
    rank_order = np.argsort(np.asarray(cv_results["predictor_rank"])[rank_rows])
    predictor_ranks = np.asarray(cv_results["predictor_rank"])[rank_rows][rank_order]
    rank_cv_mse_mean = np.asarray(cv_results["mean_response_standardized_mse"])[rank_rows][
        rank_order
    ]
    rank_cv_mse_fold_sd = np.asarray(cv_results["std_response_standardized_mse"])[rank_rows][
        rank_order
    ]

    model = PiPLSRegression(
        n_components=selected.n_components,
        predictor_rank=selected.predictor_rank,
    ).fit(X, Y)
    oof_predictions = cross_val_predict(
        model,
        X,
        Y,
        cv=KFold(n_splits=5, shuffle=False),
    )
    factors = pipls_display_factors(model.decomposition_)
    structure = latent_structure(model)
    diagnostics = prediction_diagnostics(
        Y,
        oof_predictions,
        prediction_kind="selection-conditioned OOF predictions",
    )
    return SimpleNamespace(
        X=X,
        Y=Y,
        path_search=path_search,
        component_path=component_path,
        selected=selected,
        predictor_ranks=predictor_ranks,
        rank_cv_mse_mean=rank_cv_mse_mean,
        rank_cv_mse_fold_sd=rank_cv_mse_fold_sd,
        model=model,
        oof_predictions=oof_predictions,
        factors=factors,
        structure=structure,
        diagnostics=diagnostics,
    )


def test_pulp_path_selects_the_documented_fixed_pair(pulp_result: SimpleNamespace) -> None:
    result = pulp_result

    assert isinstance(result.path_search, PiPLSPathCV)
    assert result.path_search.estimator is None
    assert result.path_search.refit is False
    assert isinstance(result.component_path, PiPLSComponentPath)
    assert result.selected.n_components == 3
    assert result.selected.predictor_rank == 10
    assert isinstance(result.model, PiPLSRegression)
    assert result.model.n_components == 3
    assert result.model.predictor_rank_ == 10


def test_pulp_rank_profile_exposes_the_upper_boundary_selection(
    pulp_result: SimpleNamespace,
) -> None:
    result = pulp_result

    np.testing.assert_array_equal(result.predictor_ranks, np.arange(3, 11))
    assert result.selected.predictor_rank == int(result.predictor_ranks[-1])
    assert result.rank_cv_mse_mean[-1] < result.rank_cv_mse_mean[-2]
    assert result.rank_cv_mse_mean[-2] - result.rank_cv_mse_mean[-1] < min(
        result.rank_cv_mse_fold_sd[-2],
        result.rank_cv_mse_fold_sd[-1],
    )


def test_pulp_oof_and_inspection_results_are_aligned(pulp_result: SimpleNamespace) -> None:
    result = pulp_result

    assert result.X.shape == (46, 14)
    assert result.Y.shape == (46, 8)
    assert result.oof_predictions.shape == result.Y.shape
    assert np.all(np.isfinite(result.oof_predictions))
    assert result.diagnostics.prediction_kind == "selection-conditioned OOF predictions"
    assert result.diagnostics.observed.shape == result.Y.shape
    assert result.factors.n_components == result.selected.n_components
    assert result.structure.n_samples == len(result.X)
    assert result.structure.n_features == result.X.shape[1]
    assert result.structure.n_targets == result.Y.shape[1]
    assert result.structure.n_components == result.selected.n_components


def test_pulp_scalar_lookup_requires_an_evaluated_component_count(
    pulp_result: SimpleNamespace,
) -> None:
    with pytest.raises(ValueError, match="was not evaluated"):
        pulp_result.component_path.for_n_components(99)
