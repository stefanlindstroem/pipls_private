"""Evaluate a standard PLSRegression component path and write a canonical CSV."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import KFold

ALGORITHM = "NIPALS"
RESULT_COLUMNS = (
    "n_components",
    "algorithm",
    "response_standardized_cv_mse_mean",
    "response_standardized_cv_mse_fold_sd",
    "n_splits",
)


def read_model_tables(x_csv: Path, y_csv: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read predictor and response CSV tables and validate their basic shape."""

    X = pd.read_csv(x_csv)
    Y = pd.read_csv(y_csv)
    if len(X) != len(Y):
        raise ValueError(f"Predictor and response row counts differ: {len(X)} != {len(Y)}.")
    if X.isna().to_numpy().any() or Y.isna().to_numpy().any():
        raise ValueError("PLS comparison tables must not contain missing values.")
    if not all(pd.api.types.is_numeric_dtype(dtype) for dtype in X.dtypes):
        raise TypeError("All predictor columns must be numeric.")
    if not all(pd.api.types.is_numeric_dtype(dtype) for dtype in Y.dtypes):
        raise TypeError("All response columns must be numeric.")
    return X, Y


def evaluate_pls_component_path(
    X: pd.DataFrame,
    Y: pd.DataFrame,
    *,
    max_n_components: int,
    n_splits: int = 5,
) -> pd.DataFrame:
    """Return fold-local response-standardized CV-MSE for standard PLS.

    One maximum-component ``PLSRegression`` fit is evaluated per fold. Because
    NIPALS extracts components sequentially, truncating its fitted rotations and
    response loadings gives the same nested component path without refitting the
    earlier components for every row.
    """

    if max_n_components < 1:
        raise ValueError("max_n_components must be at least 1.")
    algebraic_max = min(Y.shape[1], X.shape[1], X.shape[0] - 1)
    if max_n_components > algebraic_max:
        raise ValueError(
            "max_n_components exceeds the centered-data algebraic limit: "
            f"{max_n_components} > {algebraic_max}."
        )

    splitter = KFold(n_splits=n_splits, shuffle=False)
    split_mse = np.empty((max_n_components, n_splits), dtype=np.float64)
    for split_index, (train, validation) in enumerate(splitter.split(X)):
        X_train = X.iloc[train].to_numpy(dtype=np.float64)
        Y_train = Y.iloc[train].to_numpy(dtype=np.float64)
        X_validation = X.iloc[validation].to_numpy(dtype=np.float64)
        Y_validation = Y.iloc[validation].to_numpy(dtype=np.float64)

        x_mean = np.mean(X_train, axis=0)
        y_mean = np.mean(Y_train, axis=0)
        x_scale = np.std(X_train, axis=0, ddof=1)
        y_scale = np.std(Y_train, axis=0, ddof=1)
        x_scale = np.where(x_scale == 0.0, 1.0, x_scale)
        y_scale = np.where(y_scale == 0.0, 1.0, y_scale)
        if not np.all(np.isfinite(x_scale)) or not np.all(np.isfinite(y_scale)):
            raise ValueError("Training-fold scales must be finite.")

        X_train_scaled = (X_train - x_mean) / x_scale
        Y_train_scaled = (Y_train - y_mean) / y_scale
        X_validation_scaled = (X_validation - x_mean) / x_scale
        Y_validation_scaled = (Y_validation - y_mean) / y_scale

        model = PLSRegression(n_components=max_n_components, scale=False)
        model.fit(X_train_scaled, Y_train_scaled)
        for n_components in range(1, max_n_components + 1):
            x_weights = model.x_weights_[:, :n_components]
            x_loadings = model.x_loadings_[:, :n_components]
            y_loadings = model.y_loadings_[:, :n_components]
            x_rotations = x_weights @ np.linalg.pinv(x_loadings.T @ x_weights)
            prediction_scaled = X_validation_scaled @ x_rotations @ y_loadings.T
            split_mse[n_components - 1, split_index] = float(
                np.mean((Y_validation_scaled - prediction_scaled) ** 2)
            )

    rows: list[dict[str, int | float | str]] = []
    for n_components in range(1, max_n_components + 1):
        values = split_mse[n_components - 1]
        rows.append(
            {
                "n_components": n_components,
                "algorithm": ALGORITHM,
                "response_standardized_cv_mse_mean": float(np.mean(values)),
                "response_standardized_cv_mse_fold_sd": float(np.std(values)),
                "n_splits": n_splits,
            }
        )
    return pd.DataFrame(rows, columns=RESULT_COLUMNS)


def write_pls_component_path(
    x_csv: Path,
    y_csv: Path,
    output_csv: Path,
    *,
    max_n_components: int,
    n_splits: int = 5,
) -> None:
    """Evaluate standard PLS from explicit tables and write its CSV path."""

    X, Y = read_model_tables(x_csv, y_csv)
    path = evaluate_pls_component_path(
        X,
        Y,
        max_n_components=max_n_components,
        n_splits=n_splits,
    )
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    path.to_csv(output_csv, index=False)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("x_csv", type=Path)
    parser.add_argument("y_csv", type=Path)
    parser.add_argument("output_csv", type=Path)
    parser.add_argument("--max-components", type=int, required=True)
    parser.add_argument("--n-splits", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    """Evaluate one standard-PLS component path from the command line."""

    args = _parse_args()
    write_pls_component_path(
        args.x_csv,
        args.y_csv,
        args.output_csv,
        max_n_components=args.max_components,
        n_splits=args.n_splits,
    )


if __name__ == "__main__":
    main()
