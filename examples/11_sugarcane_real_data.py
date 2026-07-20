"""Compare Sugarcane component paths and export one spectral post-analysis."""

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

DATA_DIR = Path(__file__).resolve().parents[1] / "datasets" / "sugarcane"
RESULTS_DIR = Path(__file__).resolve().parent / "results"
PIPLS_PATH_CSV = RESULTS_DIR / "sugarcane_component_path.csv"
PLS_PATH_CSV = RESULTS_DIR / "sugarcane_pls_component_path.csv"
POST_ANALYSIS_DIR = RESULTS_DIR / "sugarcane_post_analysis"
CHOSEN_N_COMPONENTS = 2

X = pd.read_csv(DATA_DIR / "X.csv")
Y = pd.read_csv(DATA_DIR / "Y.csv")

# Compare Pi-PLS and ordinary PLS component paths.
path_search = PiPLSPathCV(refit=False).fit(X, Y)
pipls_path = pd.DataFrame(path_search.component_path_results_)
pipls_path.to_csv(PIPLS_PATH_CSV, index=False)
evaluate_pls_component_path(
    X,
    Y,
    max_n_components=int(path_search.n_components_values_[-1]),
).to_csv(PLS_PATH_CSV, index=False)
plot_component_path(
    PIPLS_PATH_CSV,
    PLS_PATH_CSV,
    RESULTS_DIR / "sugarcane_component_path.pdf",
    title="Sugarcane component-path comparison",
)

# Fit the selected Pi-PLS model.
chosen_predictor_rank = int(
    pipls_path.set_index("n_components").loc[CHOSEN_N_COMPONENTS, "predictor_rank"]
)
model = PiPLSRegression(
    n_components=CHOSEN_N_COMPONENTS,
    predictor_rank=chosen_predictor_rank,
).fit(X, Y)

# Export OOF predictions and fitted-model inspection tables.
oof = fixed_model_oof_predictions(
    model,
    X,
    Y,
    splitter=KFold(n_splits=5, shuffle=False),
)
write_post_analysis_tables(
    POST_ANALYSIS_DIR,
    build_post_analysis_tables(
        factors=pipls_display_factors(model.decomposition_),
        diagnostics=prediction_diagnostics(
            Y,
            oof.predictions,
            prediction_kind="selection-conditioned OOF predictions",
        ),
        structure=latent_structure(model),
        predictor_names=X.columns.tolist(),
        response_names=Y.columns.tolist(),
        sample_names=[str(index) for index in range(1, len(X) + 1)],
        fold_index=oof.fold_index,
    ),
)
render_post_analysis_report(
    POST_ANALYSIS_DIR,
    POST_ANALYSIS_DIR / "post_analysis.pdf",
    dataset_name="Sugarcane",
    predictor_style="line",
    predictor_axis=X.columns.to_numpy(dtype=float),
    predictor_axis_label="Wavelength (nm)",
    score_components=(1, 2),
    loading_components=(1, 2),
    coefficient_responses=("TS", "CP", "ADF", "IVOMD"),
)

print(f"X shape: {X.shape}; Y shape: {Y.shape}")
print(
    "Selected Pi-PLS: "
    f"n_components={model.n_components}, predictor_rank={model.predictor_rank_}"
)
print(f"Wrote results to {RESULTS_DIR}")
