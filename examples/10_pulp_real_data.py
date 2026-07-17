"""Select a Pi-PLS model after explicitly reading the pulp X and Y tables."""

from pathlib import Path

import pandas as pd

from pipls import PiPLSPathCV

DATA_DIR = Path(__file__).resolve().parents[1] / "datasets" / "pulp"

# Read predictors X and responses Y exactly as an ordinary programming user would.
# metadata.yaml documents the repository asset but is not required for model use.
X = pd.read_csv(DATA_DIR / "X.csv")
Y = pd.read_csv(DATA_DIR / "Y.csv")

expected_predictors = [
    "Shives",
    "Fines B",
    "L (arith)",
    "L (lw)",
    "L (llw)",
    "W (arith)",
    "W (lw)",
    "W (llw)",
    "C (arith)",
    "C (lw)",
    "C (llw)",
    "F (arith)",
    "F (lw)",
    "F (llw)",
]
expected_responses = [
    "CSF",
    "Density",
    "TI",
    "Elongation",
    "TEA",
    "TSI",
    "Tear index",
    "s",
]

if list(X.columns) != expected_predictors:
    raise ValueError(f"Unexpected predictor columns: {list(X.columns)!r}.")
if list(Y.columns) != expected_responses:
    raise ValueError(f"Unexpected response columns: {list(Y.columns)!r}.")
if len(X) != len(Y):
    raise ValueError(f"Predictor and response row counts differ: {len(X)} != {len(Y)}.")
if X.isna().to_numpy().any() or Y.isna().to_numpy().any():
    raise ValueError("Pulp model tables must not contain missing values.")
if not all(pd.api.types.is_numeric_dtype(dtype) for dtype in X.dtypes):
    raise TypeError("All predictor columns must be numeric.")
if not all(pd.api.types.is_numeric_dtype(dtype) for dtype in Y.dtypes):
    raise TypeError("All response columns must be numeric.")

search = PiPLSPathCV(
    n_components_values=[1, 2, 3, 4],
    search_method="auto",
    cv=5,
    return_oof_predictions=True,
).fit(X, Y)

print(f"X shape: {X.shape}")
print(f"Y shape: {Y.shape}")
print(f"selected parameters: {search.best_params_}")
print(f"selection-conditioned pooled OOF R2: {search.pooled_oof_r2_:.6f}")
