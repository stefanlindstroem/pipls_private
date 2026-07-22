"""Canonical numerical workflow shared by the Pulp example and tutorial."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline

from pipls import PiPLSComponentPath, PiPLSPathCV, PiPLSRegression
from pipls.inspection import (
    LatentStructure,
    PiPLSDisplayFactors,
    PredictionDiagnostics,
    PredictionKind,
    latent_structure,
    pipls_display_factors,
    prediction_diagnostics,
)

from .fixed_model_oof import FixedModelOOFResult, fixed_model_oof_predictions

PULP_DATA_DIR = Path(__file__).resolve().parents[2] / "datasets" / "pulp"
PULP_CHOSEN_N_COMPONENTS = 3
PULP_PREDICTION_KIND: PredictionKind = "selection-conditioned OOF predictions"


@dataclass(frozen=True)
class PulpWorkflowResult:
    """Results from the canonical Pulp model-development workflow."""

    X: pd.DataFrame
    Y: pd.DataFrame
    pipeline_template: Pipeline
    path_search: PiPLSPathCV
    component_path: PiPLSComponentPath
    chosen_n_components: int
    chosen_predictor_rank: int
    fitted_pipeline: Pipeline
    oof: FixedModelOOFResult
    factors: PiPLSDisplayFactors
    diagnostics: PredictionDiagnostics
    structure: LatentStructure

    @property
    def model(self) -> PiPLSRegression:
        """Return the fitted terminal Pi-PLS estimator."""

        return _pipls_step(self.fitted_pipeline)


# --8<-- [start:load-pulp-data]
def load_pulp_data(data_dir: Path = PULP_DATA_DIR) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read the committed Pulp predictor and response tables."""

    return pd.read_csv(data_dir / "X.csv"), pd.read_csv(data_dir / "Y.csv")
# --8<-- [end:load-pulp-data]


# --8<-- [start:build-pulp-pipeline]
def build_pulp_pipeline() -> Pipeline:
    """Return the pipeline template evaluated by the Pulp component path."""

    return Pipeline(
        [
            (
                "pipls",
                PiPLSRegression(
                    n_components=1,
                    predictor_rank=1,
                ),
            )
        ]
    )
# --8<-- [end:build-pulp-pipeline]


# --8<-- [start:evaluate-pulp-component-path]
def evaluate_pulp_component_path(
    pipeline_template: Pipeline,
    X: pd.DataFrame,
    Y: pd.DataFrame,
) -> tuple[PiPLSPathCV, PiPLSComponentPath]:
    """Evaluate the Pi-PLS component path for the complete pipeline."""

    path_search = PiPLSPathCV(
        estimator=pipeline_template,
        refit=False,
    ).fit(X, Y)
    return path_search, path_search.component_path_
# --8<-- [end:evaluate-pulp-component-path]


# --8<-- [start:select-pulp-predictor-rank]
def select_pulp_predictor_rank(
    component_path: PiPLSComponentPath,
    *,
    n_components: int,
) -> int:
    """Return the path-selected predictor rank for one component count."""

    return component_path.for_n_components(n_components).predictor_rank
# --8<-- [end:select-pulp-predictor-rank]


# --8<-- [start:fit-pulp-pipeline]
def fit_pulp_pipeline(
    pipeline_template: Pipeline,
    X: pd.DataFrame,
    Y: pd.DataFrame,
    *,
    n_components: int,
    predictor_rank: int,
) -> Pipeline:
    """Clone and fit the pipeline at one fixed Pi-PLS rank pair."""

    fitted_pipeline = clone(pipeline_template).set_params(
        pipls__n_components=n_components,
        pipls__predictor_rank=predictor_rank,
    )
    return fitted_pipeline.fit(X, Y)
# --8<-- [end:fit-pulp-pipeline]


def run_pulp_workflow(
    *,
    data_dir: Path = PULP_DATA_DIR,
    chosen_n_components: int = PULP_CHOSEN_N_COMPONENTS,
) -> PulpWorkflowResult:
    """Run path evaluation, fixed fitting, OOF prediction, and inspection."""

    X, Y = load_pulp_data(data_dir)
    pipeline_template = build_pulp_pipeline()
    path_search, component_path = evaluate_pulp_component_path(
        pipeline_template,
        X,
        Y,
    )
    chosen_predictor_rank = select_pulp_predictor_rank(
        component_path,
        n_components=chosen_n_components,
    )
    fitted_pipeline = fit_pulp_pipeline(
        pipeline_template,
        X,
        Y,
        n_components=chosen_n_components,
        predictor_rank=chosen_predictor_rank,
    )
    model = _pipls_step(fitted_pipeline)
    # --8<-- [start:pulp-oof-predictions]
    oof = fixed_model_oof_predictions(
        fitted_pipeline,
        X,
        Y,
        splitter=KFold(n_splits=5, shuffle=False),
    )
    # --8<-- [end:pulp-oof-predictions]
    # --8<-- [start:pulp-inspection-results]
    factors = pipls_display_factors(model.decomposition_)
    diagnostics = prediction_diagnostics(
        Y,
        oof.predictions,
        prediction_kind=PULP_PREDICTION_KIND,
    )
    structure = latent_structure(model)
    # --8<-- [end:pulp-inspection-results]
    return PulpWorkflowResult(
        X=X,
        Y=Y,
        pipeline_template=pipeline_template,
        path_search=path_search,
        component_path=component_path,
        chosen_n_components=chosen_n_components,
        chosen_predictor_rank=chosen_predictor_rank,
        fitted_pipeline=fitted_pipeline,
        oof=oof,
        factors=factors,
        diagnostics=diagnostics,
        structure=structure,
    )


def _pipls_step(pipeline: Pipeline) -> PiPLSRegression:
    model = pipeline.named_steps.get("pipls")
    if not isinstance(model, PiPLSRegression):
        raise TypeError("The Pulp pipeline must contain a terminal 'pipls' PiPLSRegression step.")
    return model
