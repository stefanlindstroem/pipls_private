"""Generate out-of-fold predictions for one already fixed estimator."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.base import clone
from sklearn.model_selection import BaseCrossValidator

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


@dataclass(frozen=True)
class FixedModelOOFResult:
    """Predictions and fold membership from one fixed-model CV pass."""

    predictions: FloatArray
    fold_index: IntArray
    n_splits: int


def fixed_model_oof_predictions(
    estimator: Any,
    X: Any,
    Y: Any,
    *,
    splitter: BaseCrossValidator,
) -> FixedModelOOFResult:
    """Return aligned OOF predictions without selecting model parameters.

    The estimator is cloned and fitted once per split. Every observation must
    occur in exactly one validation fold. The helper preserves pandas objects
    through row selection when they provide ``iloc``; NumPy-like inputs are
    indexed as arrays.
    """

    y_values = _response_matrix(Y)
    n_samples = y_values.shape[0]
    if len(X) != n_samples:
        raise ValueError(f"Predictor and response row counts differ: {len(X)} != {n_samples}.")

    predictions = np.full(y_values.shape, np.nan, dtype=np.float64)
    fold_index = np.zeros(n_samples, dtype=np.int64)
    split_count = 0
    for split_count, (train, validation) in enumerate(splitter.split(X, Y), start=1):
        train_index = _index_vector(train, n_samples=n_samples, name="training")
        validation_index = _index_vector(validation, n_samples=n_samples, name="validation")
        if np.intersect1d(train_index, validation_index).size:
            raise ValueError("Training and validation indices must not overlap within a fold.")
        if np.any(fold_index[validation_index] != 0):
            raise ValueError("Every observation must occur in at most one validation fold.")

        fitted = clone(estimator)
        fitted.fit(_take_rows(X, train_index), _take_rows(Y, train_index))
        fold_prediction = _response_matrix(fitted.predict(_take_rows(X, validation_index)))
        expected_shape = (validation_index.size, y_values.shape[1])
        if fold_prediction.shape != expected_shape:
            raise ValueError(
                "Estimator predictions must match the validation response shape: "
                f"expected {expected_shape}, got {fold_prediction.shape}."
            )
        predictions[validation_index] = fold_prediction
        fold_index[validation_index] = split_count

    if split_count == 0:
        raise ValueError("splitter must produce at least one split.")
    missing = np.flatnonzero(fold_index == 0)
    if missing.size:
        raise ValueError("Every observation must occur in exactly one validation fold.")
    if not np.all(np.isfinite(predictions)):
        raise ValueError("OOF predictions must contain only finite values.")

    predictions.setflags(write=False)
    fold_index.setflags(write=False)
    return FixedModelOOFResult(
        predictions=predictions,
        fold_index=fold_index,
        n_splits=split_count,
    )


def _take_rows(values: Any, indices: IntArray) -> Any:
    if hasattr(values, "iloc"):
        return values.iloc[indices]
    return np.asarray(values)[indices]


def _response_matrix(values: ArrayLike) -> FloatArray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 1:
        array = array.reshape(-1, 1)
    if array.ndim != 2:
        raise ValueError(f"Responses must be one- or two-dimensional; got shape {array.shape}.")
    if array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError("Responses must contain at least one row and one column.")
    if not np.all(np.isfinite(array)):
        raise ValueError("Responses must contain only finite values.")
    return array


def _index_vector(values: ArrayLike, *, n_samples: int, name: str) -> IntArray:
    indices = np.asarray(values)
    if indices.ndim != 1:
        raise ValueError(f"{name.capitalize()} indices must be one-dimensional.")
    if indices.size == 0:
        raise ValueError(f"Each fold must contain at least one {name} observation.")
    if not np.issubdtype(indices.dtype, np.integer):
        raise ValueError(f"{name.capitalize()} indices must be integers.")
    result = indices.astype(np.int64, copy=False)
    if np.any(result < 0) or np.any(result >= n_samples):
        raise ValueError(f"{name.capitalize()} indices are outside the sample range.")
    if np.unique(result).size != result.size:
        raise ValueError(f"{name.capitalize()} indices must not contain duplicates.")
    return result
