"""Structured cross-validation reporting for public Pi-PLS estimators."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, cast

import numpy as np
from numpy.typing import NDArray

from ._result_validation import (
    _boolean,
    _finite_float,
    _literal_string,
    _read_only_float_array,
    _read_only_int_array,
)
from .component_path import PiPLSComponentResult

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.intp]
EstimateKind = Literal["selection-conditioned", "fixed-parameter"]
_ALLOWED_ESTIMATE_KINDS = frozenset({"selection-conditioned", "fixed-parameter"})


@dataclass(frozen=True)
class PiPLSValidationReport:
    """Immutable summary of a Pi-PLS cross-validation result.

    Direct construction validates the same selected-result, array, coverage,
    and immutability invariants as reports returned by
    :class:`pipls.PiPLSSearchCV`.

    Parameters
    ----------
    selected_result : PiPLSComponentResult
        Immutable component-path result represented by the report.
    estimate_kind : {"selection-conditioned", "fixed-parameter"}
        Whether the same validation result selected model parameters or evaluated
        a parameterization fixed independently of those predictions.
    is_leave_one_out : bool
        Whether the materialized splitter is leave-one-out.
    oof_predictions : ndarray or None, default=None
        Ordered OOF predictions. One-dimensional responses produce shape
        ``(n_samples,)``; multi-output responses produce
        ``(n_samples, n_targets)``. Covered rows are finite; uncovered rows are
        represented entirely by NaN.
    oof_prediction_counts : ndarray of shape (n_samples,) or None, default=None
        Nonnegative number of validation predictions contributing to each OOF row.
    pooled_oof_r2 : float or None, default=None
        Finite pooled $R^2$ over rows with OOF coverage.

    Attributes
    ----------
    n_components : int
        Component count from ``selected_result``.
    predictor_rank : int
        Predictor rank from ``selected_result``.
    n_splits : int
        Number of cross-validation splits from ``selected_result``.
    mean_test_score : float
        Mean configured test score from ``selected_result``.
    mean_response_standardized_mse : float
        Mean response-standardized validation MSE from ``selected_result``.

    Notes
    -----
    ``estimate_kind="selection-conditioned"`` means the reported validation
    result was also used to choose ``n_components`` and/or ``predictor_rank``.
    It is therefore not an unbiased post-selection performance estimate. Arrays
    stored by the report are defensive, read-only copies.
    """

    selected_result: PiPLSComponentResult
    estimate_kind: EstimateKind
    is_leave_one_out: bool
    oof_predictions: FloatArray | None = None
    oof_prediction_counts: IntArray | None = None
    pooled_oof_r2: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.selected_result, PiPLSComponentResult):
            raise TypeError("selected_result must be a PiPLSComponentResult.")
        estimate_kind = cast(
            EstimateKind,
            _literal_string(
                self.estimate_kind,
                name="estimate_kind",
                allowed=_ALLOWED_ESTIMATE_KINDS,
            ),
        )
        is_leave_one_out = _boolean(self.is_leave_one_out, name="is_leave_one_out")
        pooled_oof_r2 = (
            None
            if self.pooled_oof_r2 is None
            else _finite_float(self.pooled_oof_r2, name="pooled_oof_r2")
        )

        predictions: FloatArray | None = None
        counts: IntArray | None = None
        if self.oof_predictions is None:
            if self.oof_prediction_counts is not None or pooled_oof_r2 is not None:
                raise ValueError("OOF counts and pooled OOF R2 require oof_predictions.")
        else:
            if self.oof_prediction_counts is None:
                raise ValueError("oof_prediction_counts are required with oof_predictions.")
            predictions = _read_only_float_array(
                self.oof_predictions,
                name="oof_predictions",
                ndim=np.asarray(self.oof_predictions).ndim,
                require_finite=False,
            )
            if predictions.ndim not in (1, 2):
                raise ValueError("oof_predictions must be one- or two-dimensional.")
            if predictions.shape[0] == 0:
                raise ValueError("oof_predictions must contain at least one row.")
            counts = _read_only_int_array(
                self.oof_prediction_counts,
                name="oof_prediction_counts",
            )
            if counts.shape[0] != predictions.shape[0]:
                raise ValueError(
                    "oof_prediction_counts must contain one value per prediction row."
                )
            if np.any(counts < 0):
                raise ValueError("oof_prediction_counts must contain nonnegative values.")
            covered = counts > 0
            uncovered = ~covered
            if pooled_oof_r2 is not None and int(np.count_nonzero(covered)) < 2:
                raise ValueError(
                    "pooled_oof_r2 requires at least two rows with OOF coverage."
                )
            if predictions.ndim == 1:
                if np.any(~np.isfinite(predictions[covered])):
                    raise ValueError("Covered OOF predictions must be finite.")
                if np.any(~np.isnan(predictions[uncovered])):
                    raise ValueError("Uncovered OOF predictions must be NaN.")
            else:
                if np.any(~np.isfinite(predictions[covered, :])):
                    raise ValueError("Covered OOF predictions must be finite.")
                if np.any(~np.isnan(predictions[uncovered, :])):
                    raise ValueError("Uncovered OOF predictions must be NaN.")

        object.__setattr__(self, "estimate_kind", estimate_kind)
        object.__setattr__(self, "is_leave_one_out", is_leave_one_out)
        object.__setattr__(self, "oof_predictions", predictions)
        object.__setattr__(self, "oof_prediction_counts", counts)
        object.__setattr__(self, "pooled_oof_r2", pooled_oof_r2)

    def __reduce__(self) -> tuple[type[PiPLSValidationReport], tuple[object, ...]]:
        """Reconstruct through validation so unpickled arrays remain read-only."""

        return (
            type(self),
            (
                self.selected_result,
                self.estimate_kind,
                self.is_leave_one_out,
                self.oof_predictions,
                self.oof_prediction_counts,
                self.pooled_oof_r2,
            ),
        )

    @property
    def n_components(self) -> int:
        """Component count represented by the report."""

        return self.selected_result.n_components

    @property
    def predictor_rank(self) -> int:
        """Predictor rank represented by the report."""

        return self.selected_result.predictor_rank

    @property
    def n_splits(self) -> int:
        """Number of cross-validation splits."""

        return self.selected_result.n_splits

    @property
    def mean_test_score(self) -> float:
        """Mean configured test score."""

        return self.selected_result.mean_test_score

    @property
    def mean_response_standardized_mse(self) -> float:
        """Mean response-standardized validation MSE."""

        return self.selected_result.cv_mse_mean

    @property
    def selection_conditioned(self) -> bool:
        """Whether the same CV result was used for parameter selection."""

        return self.estimate_kind == "selection-conditioned"

    @property
    def complete_oof_coverage(self) -> bool:
        """Whether every input row received at least one validation prediction."""

        counts = self.oof_prediction_counts
        return counts is not None and bool(np.all(counts > 0))
