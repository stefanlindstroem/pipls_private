"""Compare Pulp component paths and export one fixed-model post-analysis."""

from pathlib import Path

import pandas as pd
from _support.fixed_model_oof import fixed_model_oof_predictions
from _support.plot_component_path import plot_component_path
from _support.pls_component_path import evaluate_pls_component_path
from _support.post_analysis_artifacts import (
    build_post_analysis_tables,
    render_post_analysis_report,
    write_post_analysis_tables,
)
from sklearn.model_selection import KFold

from pipls import PiPLSPathCV, PiPLSRegression
from pipls.inspection import (
    latent_structure,
    pipls_display_factors,
    prediction_diagnostics,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPOSITORY_ROOT / "datasets" / "pulp"
RESULTS_DIR = Path(__file__).resolve().parent / "results"
COMPONENT_PATH_CSV = RESULTS_DIR / "pulp_component_path.csv"
PLS_COMPONENT_PATH_CSV = RESULTS_DIR / "pulp_pls_component_path.csv"
COMPONENT_PATH_PDF = RESULTS_DIR / "pulp_component_path.pdf"
POST_ANALYSIS_DIR = RESULTS_DIR / "pulp_post_analysis"
POST_ANALYSIS_PDF = POST_ANALYSIS_DIR / "post_analysis.pdf"

# Choose these values after inspecting the generated path CSV or PDF.
CHOSEN_N_COMPONENTS = 3
SCORE_COMPONENTS = (1, 2)
BIPLOT_COMPONENTS = (1, 2)
LOADING_COMPONENTS = (1, 2, 3)
COEFFICIENT_RESPONSES = ("CSF", "Density", "TI")
N_SPLITS = 5

X = pd.read_csv(DATA_DIR / "X.csv")
Y = pd.read_csv(DATA_DIR / "Y.csv")
predictor_names = X.columns.astype(str).tolist()
response_names = Y.columns.astype(str).tolist()
sample_names = [str(index) for index in range(1, len(X) + 1)]

# Stage 1: evaluate the component paths and write the canonical CSV files.
path_search = PiPLSPathCV(refit=False).fit(X, Y)
pipls_path = pd.DataFrame(path_search.component_path_results_)
max_n_components = int(path_search.n_components_values_[-1])
pls_path = evaluate_pls_component_path(X, Y, max_n_components=max_n_components)

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
pipls_path.to_csv(COMPONENT_PATH_CSV, index=False)
pls_path.to_csv(PLS_COMPONENT_PATH_CSV, index=False)
plot_component_path(
    COMPONENT_PATH_CSV,
    PLS_COMPONENT_PATH_CSV,
    COMPONENT_PATH_PDF,
    title="Pulp component-path comparison",
)

# Stage 2: read the chosen row and fit one fixed Pi-PLS model for interpretation.
component_path = pd.read_csv(COMPONENT_PATH_CSV).set_index("n_components")
chosen_predictor_rank = int(component_path.loc[CHOSEN_N_COMPONENTS, "predictor_rank"])
pipls_model = PiPLSRegression(
    n_components=CHOSEN_N_COMPONENTS,
    predictor_rank=chosen_predictor_rank,
).fit(X, Y)

# Stage 3: generate fixed-parameter OOF predictions and canonical post-analysis tables.
splitter = KFold(n_splits=N_SPLITS, shuffle=False)
pipls_oof = fixed_model_oof_predictions(
    PiPLSRegression(
        n_components=CHOSEN_N_COMPONENTS,
        predictor_rank=chosen_predictor_rank,
    ),
    X,
    Y,
    splitter=splitter,
)
prediction_kind = "selection-conditioned OOF predictions"
pipls_diagnostics = prediction_diagnostics(
    Y,
    pipls_oof.predictions,
    prediction_kind=prediction_kind,
)
tables = build_post_analysis_tables(
    factors=pipls_display_factors(pipls_model.decomposition_),
    diagnostics=pipls_diagnostics,
    structure=latent_structure(pipls_model),
    predictor_names=predictor_names,
    response_names=response_names,
    sample_names=sample_names,
    fold_index=pipls_oof.fold_index,
)
written_tables = write_post_analysis_tables(POST_ANALYSIS_DIR, tables)
render_post_analysis_report(
    POST_ANALYSIS_DIR,
    POST_ANALYSIS_PDF,
    dataset_name="Pulp",
    score_components=SCORE_COMPONENTS,
    biplot_components=BIPLOT_COMPONENTS,
    loading_components=LOADING_COMPONENTS,
    coefficient_responses=COEFFICIENT_RESPONSES,
)

print(f"X shape: {X.shape}")
print(f"Y shape: {Y.shape}")
print(f"Pi-PLS component-path CSV: {COMPONENT_PATH_CSV}")
print(f"PLS component-path CSV: {PLS_COMPONENT_PATH_CSV}")
print(f"comparison PDF: {COMPONENT_PATH_PDF}")
print(
    "fixed final Pi-PLS parameters: "
    f"n_components={pipls_model.n_components}, predictor_rank={pipls_model.predictor_rank_}"
)
for table_name, table_path in written_tables.items():
    print(f"{table_name} CSV: {table_path}")
print(f"post-analysis PDF: {POST_ANALYSIS_PDF}")
