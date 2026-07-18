"""Private split, scoring, and rank-search primitives for ``PiPLSPathCV``."""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any, Literal, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.metrics import r2_score
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
    n_samples: int,
    n_train_min: int,
    samples_per_predictor_rank: float,
) -> int:
    r"""Return the full-sample-supported, fold-feasible predictor-rank bound.

    The bound is

    .. math::

        \min\left(p, n_{\mathrm{train,min}} - 1,
        \left\lceil n / c \right\rceil\right),

    where ``p`` is ``n_features``, ``n`` is ``n_samples``, and ``c`` is
    ``samples_per_predictor_rank``. The smallest training-fold size remains a
    hard feasibility cap, but it does not define the statistical-support term.
    """

    _validate_positive_int(n_features, name="n_features")
    _validate_positive_int(n_samples, name="n_samples")
    _validate_positive_int(n_train_min, name="n_train_min")
    if n_train_min > n_samples:
        raise ValueError(
            "n_train_min must not exceed n_samples: "
            f"got n_train_min={n_train_min}, n_samples={n_samples}."
        )
    if n_train_min < 2:
        raise ValueError(
            "n_train_min must be at least 2 because PiPLSRegression centers each "
            f"training fold; got {n_train_min}."
        )
    samples_per_rank = _as_positive_float(
        samples_per_predictor_rank,
        name="samples_per_predictor_rank",
    )
    algebraic_limit = min(n_features, n_train_min - 1)
    if samples_per_rank <= n_samples / algebraic_limit:
        return algebraic_limit
    rule_limit = math.ceil(n_samples / samples_per_rank)
    return min(algebraic_limit, rule_limit)


def _materialize_cv_splits(
    cv: object,
    X: ArrayLike,
    y: ArrayLike,
    *,
    groups: ArrayLike | None = None,
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
        splitter.split(X_array, y_array, groups),
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


def _is_leave_one_out_splits(
    splits: tuple[CVSplit, ...],
    *,
    n_samples: int,
) -> bool:
    """Return whether splits form an ordered leave-one-out partition."""

    if len(splits) != n_samples or any(validation.size != 1 for _, validation in splits):
        return False
    counts = np.zeros(n_samples, dtype=np.intp)
    for _, validation in splits:
        counts[validation] += 1
    return bool(np.all(counts == 1))


def _validate_singleton_fold_scoring(
    scoring: object,
    splits: tuple[CVSplit, ...],
) -> None:
    """Reject ordinary R2 scoring when any validation fold is a singleton."""

    if not any(validation.size == 1 for _, validation in splits):
        return
    score_func = getattr(scoring, "_score_func", None)
    if scoring is None or scoring == "r2" or score_func is r2_score:
        raise ValueError(
            "R2 scoring is undefined for singleton validation folds. Use "
            "neg_response_standardized_mean_squared_error or another "
            "singleton-safe scorer, and compute pooled OOF R2 only as a "
            "secondary diagnostic."
        )


def _pooled_oof_r2(
    y: ArrayLike,
    predictions: ArrayLike,
    counts: ArrayLike,
) -> float | None:
    """Return pooled R2 on rows with OOF coverage, or None if unavailable."""

    y_array = np.asarray(y, dtype=np.float64)
    prediction_array = np.asarray(predictions, dtype=np.float64)
    count_array = np.asarray(counts, dtype=np.intp)
    covered = count_array > 0
    if int(np.sum(covered)) < 2:
        return None
    if y_array.ndim == 1 and prediction_array.ndim == 2:
        prediction_array = prediction_array[:, 0]
    return float(
        r2_score(
            y_array[covered],
            prediction_array[covered],
            multioutput="uniform_average",
        )
    )


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


def _search_predictor_ranks(
    *,
    allowed_ranks: ArrayLike,
    search_method: Literal["optimal", "auto"],
    evaluate: Callable[[IntArray], IntArray],
    evaluated_scores: Callable[[], tuple[IntArray, FloatArray]],
) -> tuple[IntArray, ...]:
    """Return the evaluated batches from one rank search."""

    allowed = np.asarray(allowed_ranks)
    if allowed.ndim != 1 or allowed.size == 0 or allowed.dtype.kind not in "iu":
        raise ValueError("allowed_ranks must be a nonempty one-dimensional integer array.")
    allowed = np.asarray(np.unique(allowed), dtype=np.intp)
    if np.any(allowed < 1):
        raise ValueError("allowed_ranks must contain positive integers.")
    if search_method not in ("optimal", "auto"):
        raise ValueError('search_method must be "optimal" or "auto".')

    history: list[IntArray] = []
    interval = allowed
    while True:
        if search_method == "optimal" or interval.size <= _ADAPTIVE_EXHAUSTIVE_THRESHOLD:
            proposed = interval
        else:
            logarithmic = _logarithmic_predictor_rank_values(
                lower=int(interval[0]),
                upper=int(interval[-1]),
            )
            indices = np.abs(interval[:, None] - logarithmic[None, :]).argmin(axis=0)
            proposed = np.asarray(np.unique(interval[indices]), dtype=np.intp)

        evaluated = evaluate(proposed)
        if evaluated.size:
            history.append(evaluated.copy())

        if search_method == "optimal" or interval.size <= _ADAPTIVE_EXHAUSTIVE_THRESHOLD:
            break

        ranks, scores = evaluated_scores()
        lower, upper = _adaptive_refinement_interval(ranks, -scores)
        refined = allowed[(allowed >= lower) & (allowed <= upper)]
        if np.array_equal(refined, interval):
            evaluated_set = {int(rank) for rank in ranks}
            remaining = np.asarray(
                [rank for rank in interval if int(rank) not in evaluated_set],
                dtype=np.intp,
            )
            evaluated = evaluate(remaining)
            if evaluated.size:
                history.append(evaluated.copy())
            break
        interval = refined

    return tuple(history)


def _rank_test_scores(
    mean_scores: ArrayLike,
    *,
    rtol: float = _SELECTION_RTOL,
    atol: float = _SELECTION_ATOL,
) -> IntArray:
    """Return scikit-learn-style minimum ranks with tolerant score ties."""

    scores = np.asarray(mean_scores, dtype=np.float64)
    if scores.ndim != 1 or scores.size == 0 or not np.all(np.isfinite(scores)):
        raise ValueError("mean_scores must be a nonempty finite one-dimensional array.")
    order = np.argsort(-scores, kind="mergesort")
    ranks = np.empty(scores.size, dtype=np.intp)
    group_start = 0
    for position, index in enumerate(order):
        if position > 0 and not np.isclose(
            scores[index],
            scores[order[position - 1]],
            rtol=rtol,
            atol=atol,
        ):
            group_start = position
        ranks[index] = group_start + 1
    return ranks


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
