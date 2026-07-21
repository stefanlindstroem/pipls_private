from __future__ import annotations

import importlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.pipeline import Pipeline

from pipls import PiPLSPathCV, PiPLSRegression


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _workflow_module() -> object:
    examples_dir = _repository_root() / "examples"
    path_entry = str(examples_dir)
    if path_entry not in sys.path:
        sys.path.insert(0, path_entry)
    return importlib.import_module("_support.pulp_workflow")


WORKFLOW = _workflow_module()


@pytest.fixture(scope="module")
def pulp_result() -> object:
    return WORKFLOW.run_pulp_workflow()


def test_pulp_pipeline_is_a_cloneable_terminal_pipls_pipeline() -> None:
    pipeline = WORKFLOW.build_pulp_pipeline()

    assert isinstance(pipeline, Pipeline)
    assert list(pipeline.named_steps) == ["pipls"]
    assert isinstance(pipeline.named_steps["pipls"], PiPLSRegression)
    assert pipeline.named_steps["pipls"].n_components == 1
    assert pipeline.named_steps["pipls"].predictor_rank == 1
    assert isinstance(clone(pipeline), Pipeline)


def test_pulp_workflow_evaluates_the_pipeline_and_transfers_the_selected_pair(
    pulp_result: object,
) -> None:
    result = pulp_result

    assert isinstance(result.path_search, PiPLSPathCV)
    assert isinstance(result.path_search.estimator, Pipeline)
    assert result.path_search.refit is False
    assert set(result.path_search.best_params_) == {
        "pipls__n_components",
        "pipls__predictor_rank",
    }
    assert tuple(result.component_path.columns) == (
        "n_components",
        "predictor_rank",
        "predictor_rank_policy",
        "response_standardized_cv_mse_mean",
        "response_standardized_cv_mse_fold_sd",
        "n_splits",
    )
    assert result.chosen_n_components == WORKFLOW.PULP_CHOSEN_N_COMPONENTS == 3
    assert result.chosen_predictor_rank == WORKFLOW.select_pulp_predictor_rank(
        result.component_path,
        n_components=result.chosen_n_components,
    )
    assert isinstance(result.fitted_pipeline, Pipeline)
    assert result.fitted_pipeline is not result.pipeline_template
    assert not hasattr(result.pipeline_template.named_steps["pipls"], "coef_")
    assert result.model.n_components == result.chosen_n_components
    assert result.model.predictor_rank_ == result.chosen_predictor_rank


def test_pulp_workflow_aligns_oof_and_inspection_outputs(pulp_result: object) -> None:
    result = pulp_result

    assert result.X.shape == (46, 14)
    assert result.Y.shape == (46, 8)
    assert result.oof.predictions.shape == result.Y.shape
    assert result.oof.fold_index.shape == (len(result.X),)
    assert set(result.oof.fold_index) == {1, 2, 3, 4, 5}
    assert np.all(np.isfinite(result.oof.predictions))
    assert result.diagnostics.prediction_kind == WORKFLOW.PULP_PREDICTION_KIND
    assert result.diagnostics.observed.shape == result.Y.shape
    assert result.factors.n_components == result.chosen_n_components
    assert result.structure.n_samples == len(result.X)
    assert result.structure.n_features == result.X.shape[1]
    assert result.structure.n_targets == result.Y.shape[1]
    assert result.structure.n_components == result.chosen_n_components


def test_pulp_rank_selection_requires_one_admissible_path_row() -> None:
    duplicate = pd.DataFrame(
        {
            "n_components": [2, 2],
            "predictor_rank": [3, 4],
        }
    )
    with pytest.raises(ValueError, match="exactly one row"):
        WORKFLOW.select_pulp_predictor_rank(duplicate, n_components=2)

    inadmissible = pd.DataFrame(
        {
            "n_components": [3],
            "predictor_rank": [2],
        }
    )
    with pytest.raises(ValueError, match="must not be smaller"):
        WORKFLOW.select_pulp_predictor_rank(inadmissible, n_components=3)
