"""Scoring utilities for Pi-PLS model selection."""

from __future__ import annotations

from typing import Any

from numpy.typing import ArrayLike
from sklearn.utils.validation import check_is_fitted

from .model_selection import _response_standardized_mse


def response_standardized_mean_squared_error(
    estimator: Any,
    X: ArrayLike,
    y: ArrayLike,
) -> float:
    """Return response-standardized MSE using scales learned by ``estimator``.

    The callable follows the scikit-learn scorer protocol ``(estimator, X, y)``.
    Response scales are sample standard deviations estimated from the data used
    to fit the estimator, independently of whether estimator preprocessing uses
    response scaling.
    """

    check_is_fitted(estimator, attributes=["response_scale_for_scoring_"])
    return _response_standardized_mse(
        y,
        estimator.predict(X),
        estimator.response_scale_for_scoring_,
    )


def neg_response_standardized_mean_squared_error(
    estimator: Any,
    X: ArrayLike,
    y: ArrayLike,
) -> float:
    """Return negative response-standardized MSE for scorer maximization."""

    return -response_standardized_mean_squared_error(estimator, X, y)
