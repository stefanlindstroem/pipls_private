"""Fit Pi-PLS after explicitly reading the Linnerud X and Y tables."""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import LeaveOneOut

from pipls import PiPLSRegression

DATA_DIR = Path(__file__).resolve().parents[1] / "datasets" / "linnerud"

# Read predictors X and responses Y exactly as an ordinary programming user would.
# metadata.yaml documents the repository dataset but is not needed for model use.
X = pd.read_csv(DATA_DIR / "X.csv")
Y = pd.read_csv(DATA_DIR / "Y.csv")

expected_predictors = ["Chins", "Situps", "Jumps"]
expected_responses = ["Weight", "Waist", "Pulse"]

if list(X.columns) != expected_predictors:
    raise ValueError(f"Unexpected predictor columns: {list(X.columns)!r}.")
if list(Y.columns) != expected_responses:
    raise ValueError(f"Unexpected response columns: {list(Y.columns)!r}.")
if len(X) != len(Y):
    raise ValueError(f"Predictor and response row counts differ: {len(X)} != {len(Y)}.")
if X.isna().to_numpy().any() or Y.isna().to_numpy().any():
    raise ValueError("Linnerud model tables must not contain missing values.")
if not all(pd.api.types.is_numeric_dtype(dtype) for dtype in X.dtypes):
    raise TypeError("All predictor columns must be numeric.")
if not all(pd.api.types.is_numeric_dtype(dtype) for dtype in Y.dtypes):
    raise TypeError("All response columns must be numeric.")

model = PiPLSRegression(
    n_components=2,
    predictor_rank="optimal",
    samples_per_predictor_rank=5,
    cv=LeaveOneOut(),
    return_oof_predictions=True,
).fit(X, Y)

print(f"X shape: {X.shape}")
print(f"Y shape: {Y.shape}")
print(f"selected predictor rank: {model.predictor_rank_}")
print(f"selection-conditioned pooled OOF R2: {model.pooled_oof_r2_:.6f}")
