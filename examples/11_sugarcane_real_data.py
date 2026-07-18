"""Scan the Sugarcane component path, then fit one chosen Pi-PLS model."""

import subprocess
import sys
from pathlib import Path

import pandas as pd

from pipls import PiPLSPathCV, PiPLSRegression

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPOSITORY_ROOT / "datasets" / "sugarcane"
RESULTS_DIR = Path(__file__).resolve().parent / "results"
COMPONENT_PATH_CSV = RESULTS_DIR / "sugarcane_component_path.csv"
COMPONENT_PATH_PDF = RESULTS_DIR / "sugarcane_component_path.pdf"
N_COMPONENTS_VALUES = (1, 2, 3, 4)

# This example uses three components as an explicit parsimonious choice after
# inspecting the path. Change the constant to fit another recorded row.
CHOSEN_N_COMPONENTS = 2

# Read predictors X and responses Y exactly as an ordinary programming user would.
# metadata.yaml documents the repository asset but is not required for model use.
X = pd.read_csv(DATA_DIR / "X.csv")
Y = pd.read_csv(DATA_DIR / "Y.csv")

if len(X) != len(Y):
    raise ValueError(f"Predictor and response row counts differ: {len(X)} != {len(Y)}.")
if X.isna().to_numpy().any() or Y.isna().to_numpy().any():
    raise ValueError("Sugarcane model tables must not contain missing values.")
if not all(pd.api.types.is_numeric_dtype(dtype) for dtype in X.dtypes):
    raise TypeError("All predictor columns must be numeric.")
if not all(pd.api.types.is_numeric_dtype(dtype) for dtype in Y.dtypes):
    raise TypeError("All response columns must be numeric.")

# Stage 1: scan n_components. Predictor rank is selected conditionally for each row.
path_search = PiPLSPathCV(
    n_components_values=N_COMPONENTS_VALUES,
    refit=False,
).fit(X, Y)

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
pd.DataFrame(path_search.component_path_results_).to_csv(
    COMPONENT_PATH_CSV,
    index=False,
)
subprocess.run(
    [
        sys.executable,
        str(Path(__file__).with_name("plot_component_path.py")),
        str(COMPONENT_PATH_CSV),
        str(COMPONENT_PATH_PDF),
        "--title",
        "Sugarcane Pi-PLS component path",
    ],
    check=True,
)

# Stage 2: read the canonical CSV and fit the chosen fixed parameterization.
component_path = pd.read_csv(COMPONENT_PATH_CSV)
chosen_rows = component_path.loc[component_path["n_components"] == CHOSEN_N_COMPONENTS]
if len(chosen_rows) != 1:
    raise ValueError(f"Expected one component-path row for n_components={CHOSEN_N_COMPONENTS}.")
chosen_predictor_rank = int(chosen_rows.iloc[0]["predictor_rank"])
model = PiPLSRegression(
    n_components=CHOSEN_N_COMPONENTS,
    predictor_rank=chosen_predictor_rank,
).fit(X, Y)

print(f"X shape: {X.shape}")
print(f"Y shape: {Y.shape}")
print(f"component-path CSV: {COMPONENT_PATH_CSV}")
print(f"component-path PDF: {COMPONENT_PATH_PDF}")
print(
    "fixed final parameters: "
    f"n_components={model.n_components}, predictor_rank={model.predictor_rank_}"
)
