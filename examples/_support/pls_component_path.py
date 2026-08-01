"""Evaluate a standard PLS component path for comparison with Pi-PLS."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import KFold

ALGORITHM = "NIPALS"
FloatArray = NDArray[np.float64]
IntArray = NDArray[np.intp]


@dataclass(frozen=True)
class PLSComponentPath:
    """Immutable ordinary-PLS component-path results.

    Arrays contain one result per evaluated component count and are defensive,
    read-only copies aligned by row.
    """

    n_components: IntArray
    cv_mse_mean: FloatArray
    cv_mse_fold_sd: FloatArray
    algorithm: str
    n_splits: int

    def __post_init__(self) -> None:
        n_components = _read_only_int_array(self.n_components, name="n_components")
        cv_mse_mean = _read_only_float_array(self.cv_mse_mean, name="cv_mse_mean")
        cv_mse_fold_sd = _read_only_float_array(
            self.cv_mse_fold_sd,
            name="cv_mse_fold_sd",
        )
        if n_components.size == 0:
            raise ValueError("A PLS component path must contain at least one result.")
        if cv_mse_mean.size != n_components.size or cv_mse_fold_sd.size != n_components.size:
            raise ValueError("All PLS component-path arrays must have the same length.")
        if np.any(n_components <= 0):
            raise ValueError("n_components must contain positive integers.")
        if np.any(np.diff(n_components) <= 0):
            raise ValueError("n_components must be unique and strictly ascending.")
        if not np.isfinite(cv_mse_mean).all():
            raise ValueError("cv_mse_mean must contain only finite values.")
        if not np.isfinite(cv_mse_fold_sd).all() or np.any(cv_mse_fold_sd < 0.0):
            raise ValueError("cv_mse_fold_sd must contain finite nonnegative values.")
        algorithm = str(self.algorithm).strip()
        if not algorithm:
            raise ValueError("algorithm must be a nonempty string.")
        if isinstance(self.n_splits, bool) or not isinstance(self.n_splits, (int, np.integer)):
            raise ValueError("n_splits must be a positive integer.")
        n_splits = int(self.n_splits)
        if n_splits <= 0:
            raise ValueError("n_splits must be a positive integer.")

        object.__setattr__(self, "n_components", n_components)
        object.__setattr__(self, "cv_mse_mean", cv_mse_mean)
        object.__setattr__(self, "cv_mse_fold_sd", cv_mse_fold_sd)
        object.__setattr__(self, "algorithm", algorithm)
        object.__setattr__(self, "n_splits", n_splits)

    @property
    def cv_mse_standard_error(self) -> FloatArray:
        """Return read-only fold-based standard errors of mean CV-MSE."""

        if self.n_splits < 2:
            raise ValueError(
                "cv_mse_standard_error requires at least two validation splits."
            )
        standard_error = np.asarray(
            self.cv_mse_fold_sd / np.sqrt(self.n_splits - 1),
            dtype=np.float64,
        )
        standard_error.setflags(write=False)
        return standard_error

    def __reduce__(self) -> tuple[type[PLSComponentPath], tuple[object, ...]]:
        """Reconstruct through validation so unpickled arrays remain read-only."""

        return (
            type(self),
            (
                self.n_components,
                self.cv_mse_mean,
                self.cv_mse_fold_sd,
                self.algorithm,
                self.n_splits,
            ),
        )


def evaluate_pls_component_path(
    X: pd.DataFrame,
    Y: pd.DataFrame,
    *,
    max_n_components: int,
    cv: KFold,
) -> PLSComponentPath:
    """Return fold-local response-standardized CV-MSE for standard PLS.

    One maximum-component ``PLSRegression`` fit is evaluated per fold. NIPALS
    extracts components sequentially, so truncating the fitted rotations and
    response loadings gives the same nested path without repeated earlier fits.
    """

    if len(X) != len(Y):
        raise ValueError(f"Predictor and response row counts differ: {len(X)} != {len(Y)}.")
    if max_n_components < 1:
        raise ValueError("max_n_components must be at least 1.")
    algebraic_max = min(Y.shape[1], X.shape[1], X.shape[0] - 1)
    if max_n_components > algebraic_max:
        raise ValueError(
            "max_n_components exceeds the centered-data algebraic limit: "
            f"{max_n_components} > {algebraic_max}."
        )

    n_splits = cv.get_n_splits(X, Y)
    split_mse = np.empty((max_n_components, n_splits), dtype=np.float64)
    for split_index, (train, validation) in enumerate(cv.split(X, Y)):
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

    return PLSComponentPath(
        n_components=np.arange(1, max_n_components + 1, dtype=np.intp),
        cv_mse_mean=np.mean(split_mse, axis=1),
        cv_mse_fold_sd=np.std(split_mse, axis=1),
        algorithm=ALGORITHM,
        n_splits=n_splits,
    )


def _read_only_int_array(value: ArrayLike, *, name: str) -> IntArray:
    array = np.array(value, dtype=np.intp, copy=True)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional; got shape {array.shape}.")
    array.setflags(write=False)
    return array


def _read_only_float_array(value: ArrayLike, *, name: str) -> FloatArray:
    array = np.array(value, dtype=np.float64, copy=True)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional; got shape {array.shape}.")
    array.setflags(write=False)
    return array
