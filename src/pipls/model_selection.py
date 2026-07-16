"""Private model-selection primitives shared by Pi-PLS selection modes."""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.model_selection import check_cv

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.intp]
CVSplit = tuple[IntArray, IntArray]

_SELECTION_RTOL = 1e-12
_SELECTION_ATOL = 1e-15
_ADAPTIVE_INITIAL_POINTS = 7
_ADAPTIVE_EXHAUSTIVE_THRESHOLD = 10


@dataclass(frozen=True)
class _MaterializedCV:
    """Reusable validated cross-validation splits."""

    splits: tuple[CVSplit, ...]
    n_train_min: int


def _max_predictor_rank(
    *,
    n_features: int,
    n_train_min: int,
    samples_per_predictor_rank: float,
) -> int:
    r"""Return the fold-safe upper predictor-rank bound.

    The bound is

    .. math::

        \min\left(p, n_{\mathrm{train,min}},
        \left\lceil n_{\mathrm{train,min}} / c \right\rceil\right),

    where ``p`` is ``n_features`` and ``c`` is
    ``samples_per_predictor_rank``.
    """

    _validate_positive_int(n_features, name="n_features")
    _validate_positive_int(n_train_min, name="n_train_min")
    samples_per_rank = _as_positive_float(
        samples_per_predictor_rank,
        name="samples_per_predictor_rank",
    )
    rule_limit = math.ceil(n_train_min / samples_per_rank)
    return min(n_features, n_train_min, rule_limit)


def _materialize_cv_splits(
    cv: object,
    X: ArrayLike,
    y: ArrayLike,
) -> _MaterializedCV:
    """Materialize one validated set of CV splits for reuse by all candidates."""

    X_array = np.asarray(X)
    y_array = np.asarray(y)
    if X_array.ndim != 2:
        raise ValueError(f"X must be two-dimensional; got shape {X_array.shape}.")
    if y_array.ndim not in (1, 2):
        raise ValueError(f"y must be one- or two-dimensional; got shape {y_array.shape}.")
    if X_array.shape[0] != y_array.shape[0]:
        raise ValueError(
            "X and y must contain the same number of samples: "
            f"got {X_array.shape[0]} and {y_array.shape[0]}."
        )

    splitter = check_cv(cv=cv, y=y_array, classifier=False)
    raw_splits = cast(
        Iterable[tuple[ArrayLike, ArrayLike]],
        splitter.split(X_array, y_array),
    )
    splits: list[CVSplit] = []
    n_samples = int(X_array.shape[0])
    for split_index, (train, validation) in enumerate(raw_splits):
        train_index = _as_index_array(
            train,
            name=f"training indices for split {split_index}",
            n_samples=n_samples,
        )
        validation_index = _as_index_array(
            validation,
            name=f"validation indices for split {split_index}",
            n_samples=n_samples,
        )
        if np.intersect1d(train_index, validation_index).size:
            raise ValueError(f"Training and validation indices overlap in split {split_index}.")
        splits.append((train_index.copy(), validation_index.copy()))

    if not splits:
        raise ValueError("Cross-validation must produce at least one split.")
    return _MaterializedCV(
        splits=tuple(splits),
        n_train_min=min(train.size for train, _ in splits),
    )


def _predictor_rank_values(*, n_components: int, max_predictor_rank: int) -> IntArray:
    """Return every admissible predictor rank for one fixed component count."""

    _validate_positive_int(n_components, name="n_components")
    _validate_positive_int(max_predictor_rank, name="max_predictor_rank")
    if n_components > max_predictor_rank:
        raise ValueError(
            "n_components must not exceed max_predictor_rank: "
            f"got {n_components} and {max_predictor_rank}."
        )
    return np.arange(n_components, max_predictor_rank + 1, dtype=np.intp)


def _logarithmic_predictor_rank_values(
    *,
    lower: int,
    upper: int,
    n_values: int = _ADAPTIVE_INITIAL_POINTS,
) -> IntArray:
    """Return deterministic approximately logarithmic integer ranks including endpoints."""

    _validate_positive_int(lower, name="lower")
    _validate_positive_int(upper, name="upper")
    _validate_positive_int(n_values, name="n_values")
    if lower > upper:
        raise ValueError(f"lower must not exceed upper: got {lower} and {upper}.")
    if lower == upper:
        return np.asarray([lower], dtype=np.intp)

    values = np.exp(np.linspace(math.log(lower), math.log(upper), n_values))
    rounded = np.rint(values).astype(np.intp)
    unique_values: IntArray = np.unique(
        np.concatenate(
            (
                np.asarray([lower], dtype=np.intp),
                rounded,
                np.asarray([upper], dtype=np.intp),
            )
        )
    )
    return unique_values


def _adaptive_refinement_interval(
    predictor_ranks: ArrayLike,
    mean_losses: ArrayLike,
) -> tuple[int, int]:
    """Return the evaluated-neighbor interval around the current best rank."""

    ranks = np.asarray(predictor_ranks)
    losses = np.asarray(mean_losses, dtype=np.float64)
    selected_rank, _ = _select_predictor_rank(ranks, losses)
    order = np.argsort(ranks)
    sorted_ranks = ranks[order]
    selected_index = int(np.flatnonzero(sorted_ranks == selected_rank)[0])
    lower_index = max(0, selected_index - 1)
    upper_index = min(sorted_ranks.size - 1, selected_index + 1)
    return int(sorted_ranks[lower_index]), int(sorted_ranks[upper_index])


def _training_response_scale(y_train: ArrayLike) -> FloatArray:
    """Return fold-local response scales using ``ddof=1`` and unit zero scales."""

    y_array = _as_2d_targets(y_train, name="y_train")
    if y_array.shape[0] <= 1:
        return np.ones(y_array.shape[1], dtype=np.float64)
    scale = np.std(y_array, axis=0, ddof=1)
    return np.where(scale == 0.0, 1.0, scale).astype(np.float64, copy=False)


def _response_standardized_mse(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    response_scale: ArrayLike,
) -> float:
    """Return uniformly response-weighted MSE after fold-local scaling."""

    true_array = _as_2d_targets(y_true, name="y_true")
    pred_array = _as_2d_targets(y_pred, name="y_pred")
    if true_array.shape != pred_array.shape:
        raise ValueError(
            "y_true and y_pred must have identical shapes: "
            f"got {true_array.shape} and {pred_array.shape}."
        )
    scale = np.asarray(response_scale, dtype=np.float64)
    if scale.ndim != 1 or scale.shape[0] != true_array.shape[1]:
        raise ValueError(
            "response_scale must contain one value per response: "
            f"expected {(true_array.shape[1],)}, got {scale.shape}."
        )
    if not np.all(np.isfinite(scale)) or np.any(scale <= 0.0):
        raise ValueError("response_scale must contain positive finite values.")
    standardized_residual = (true_array - pred_array) / scale[None, :]
    return float(np.mean(np.square(standardized_residual)))


def _select_predictor_rank(
    predictor_ranks: ArrayLike,
    mean_losses: ArrayLike,
    *,
    rtol: float = _SELECTION_RTOL,
    atol: float = _SELECTION_ATOL,
) -> tuple[int, float]:
    """Select the smallest rank whose loss ties the minimum within tolerance."""

    ranks = np.asarray(predictor_ranks)
    losses = np.asarray(mean_losses, dtype=np.float64)
    if ranks.ndim != 1 or ranks.size == 0:
        raise ValueError("predictor_ranks must be a nonempty one-dimensional array.")
    if ranks.dtype.kind not in "iu" or np.any(ranks < 1):
        raise ValueError("predictor_ranks must contain positive integers.")
    if np.unique(ranks).size != ranks.size:
        raise ValueError("predictor_ranks must not contain duplicates.")
    if losses.ndim != 1 or losses.shape != ranks.shape:
        raise ValueError("mean_losses must be one-dimensional with one value per predictor rank.")
    if not np.all(np.isfinite(losses)):
        raise ValueError("mean_losses must contain only finite values.")
    if not np.isfinite(rtol) or rtol < 0.0 or not np.isfinite(atol) or atol < 0.0:
        raise ValueError("rtol and atol must be nonnegative finite values.")

    minimum_loss = float(np.min(losses))
    tied = np.isclose(losses, minimum_loss, rtol=rtol, atol=atol)
    selected_rank = int(np.min(ranks[tied]))
    selected_index = int(np.flatnonzero(ranks == selected_rank)[0])
    return selected_rank, float(losses[selected_index])


def _as_index_array(index: ArrayLike, *, name: str, n_samples: int) -> IntArray:
    array = np.asarray(index)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional; got shape {array.shape}.")
    if array.size == 0:
        raise ValueError(f"{name} must not be empty.")
    if array.dtype.kind not in "iu":
        raise ValueError(f"{name} must contain integer indices.")
    converted = np.asarray(array, dtype=np.intp)
    if np.unique(converted).size != converted.size:
        raise ValueError(f"{name} must not contain duplicate indices.")
    if np.any(converted < 0) or np.any(converted >= n_samples):
        raise ValueError(f"{name} contains an index outside [0, {n_samples}).")
    return converted


def _as_2d_targets(values: ArrayLike, *, name: str) -> FloatArray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 1:
        array = array.reshape(-1, 1)
    if array.ndim != 2:
        raise ValueError(f"{name} must be one- or two-dimensional; got shape {array.shape}.")
    if array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError(f"{name} must contain at least one sample and one response.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values.")
    return array


def _validate_positive_int(value: Any, *, name: str) -> None:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be a positive integer; got {value!r}.")
    if int(value) < 1:
        raise ValueError(f"{name} must be at least 1; got {value!r}.")


def _as_positive_float(value: Any, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value,
        (int, float, np.integer, np.floating),
    ):
        raise ValueError(f"{name} must be a positive finite number; got {value!r}.")
    numeric_value = float(value)
    if not np.isfinite(numeric_value) or numeric_value <= 0.0:
        raise ValueError(f"{name} must be a positive finite number; got {value!r}.")
    return numeric_value
