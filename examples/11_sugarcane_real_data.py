"""Fit Pi-PLS after explicitly reading the sugarcane X and Y tables."""

from pathlib import Path

import pandas as pd

from pipls import PiPLSRegression

DATA_DIR = Path(__file__).resolve().parents[1] / "datasets" / "sugarcane"

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

model = PiPLSRegression(
    n_components=2,
    predictor_rank=8,
    svd_solver="randomized",
    random_state=0,
).fit(X, Y)

print(f"X shape: {X.shape}")
print(f"Y shape: {Y.shape}")
print(f"predictor rank: {model.predictor_rank_}")
print(f"training R2: {model.score(X, Y):.6f}")
