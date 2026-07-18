"""Compare Tobacco Pi-PLS and PLS paths, then fit one Pi-PLS model."""

from pathlib import Path

import pandas as pd
from plot_component_path import plot_component_path
from pls_component_path import evaluate_pls_component_path

from pipls import PiPLSPathCV, PiPLSRegression

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPOSITORY_ROOT / "datasets" / "tobacco"
RESULTS_DIR = Path(__file__).resolve().parent / "results"
COMPONENT_PATH_CSV = RESULTS_DIR / "tobacco_component_path.csv"
PLS_COMPONENT_PATH_CSV = RESULTS_DIR / "tobacco_pls_component_path.csv"
COMPONENT_PATH_PDF = RESULTS_DIR / "tobacco_component_path.pdf"

# Choose this value after inspecting the generated CSV or PDF.
CHOSEN_N_COMPONENTS = 8

X = pd.read_csv(DATA_DIR / "X.csv")
Y = pd.read_csv(DATA_DIR / "Y.csv")

# Stage 1: scan predictor rank conditionally for every component count. Full
# predictor SVD is explicit because randomized SVD has a dedicated benchmark.
path_search = PiPLSPathCV(
    estimator=PiPLSRegression(svd_solver="full"),
    search_method="auto",
    refit=False,
    n_jobs=1,
).fit(X, Y)
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
    title="Tobacco component-path comparison",
)

# Stage 2: read the chosen Pi-PLS row and fit the same fixed full-SVD model.
component_path = pd.read_csv(COMPONENT_PATH_CSV).set_index("n_components")
chosen_predictor_rank = int(component_path.loc[CHOSEN_N_COMPONENTS, "predictor_rank"])
model = PiPLSRegression(
    n_components=CHOSEN_N_COMPONENTS,
    predictor_rank=chosen_predictor_rank,
    svd_solver="full",
).fit(X, Y)

print(f"X shape: {X.shape}")
print(f"Y shape: {Y.shape}")
print(f"Pi-PLS component-path CSV: {COMPONENT_PATH_CSV}")
print(f"PLS component-path CSV: {PLS_COMPONENT_PATH_CSV}")
print(f"comparison PDF: {COMPONENT_PATH_PDF}")
print("path search method: auto")
print(f"fixed-model predictor SVD: {model.decomposition_.predictor_svd_solver}")
print(
    "fixed final Pi-PLS parameters: "
    f"n_components={model.n_components}, predictor_rank={model.predictor_rank_}"
)
