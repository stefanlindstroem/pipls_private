"""Create a Pulp Pi-PLS component path and fixed-model post-analysis."""

from pathlib import Path

import pandas as pd
from _support.fixed_model_oof import fixed_model_oof_predictions
from _support.plot_component_path import plot_pipls_component_path
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

DATA_DIR = Path(__file__).resolve().parents[1] / "datasets" / "pulp"
ANALYSIS_DIR = Path(__file__).resolve().parent / "results" / "pulp_post_analysis"
CHOSEN_N_COMPONENTS = 3

X = pd.read_csv(DATA_DIR / "X.csv")
Y = pd.read_csv(DATA_DIR / "Y.csv")

# Evaluate the Pi-PLS component path.
path_search = PiPLSPathCV(refit=False).fit(X, Y)
pipls_path = pd.DataFrame(path_search.component_path_results_)
pipls_path.to_csv(ANALYSIS_DIR / "component_path.csv", index=False)
plot_pipls_component_path(
    ANALYSIS_DIR / "component_path.csv",
    ANALYSIS_DIR / "component_path.pdf",
    title="Pulp Pi-PLS component path",
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
    ANALYSIS_DIR,
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
    ANALYSIS_DIR,
    ANALYSIS_DIR / "post_analysis.pdf",
    dataset_name="Pulp",
    score_components=(1, 2),
    biplot_components=(1, 2),
    loading_components=(1, 2, 3),
    coefficient_responses=("CSF", "Density", "TI"),
)

print(f"X shape: {X.shape}; Y shape: {Y.shape}")
print(
    "Selected Pi-PLS: "
    f"n_components={model.n_components}, predictor_rank={model.predictor_rank_}"
)
print(f"Wrote results to {ANALYSIS_DIR}")
