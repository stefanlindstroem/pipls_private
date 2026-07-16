"""Predictor-rank bounds shared by Pi-PLS selection modes."""

from __future__ import annotations

import math
from typing import Any

import numpy as np


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
