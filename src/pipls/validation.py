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
    _nonnegative_finite_float,
    _positive_int,
    _read_only_float_array,
    _read_only_int_array,
)

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.intp]
EstimateKind = Literal["selection-conditioned", "fixed-parameter"]
_ALLOWED_ESTIMATE_KINDS = frozenset({"selection-conditioned", "fixed-parameter"})


@dataclass(frozen=True)
class PiPLSValidationReport:
    """Immutable summary of a Pi-PLS cross-validation result.

    Direct construction validates the same scalar, array, coverage, and
    immutability invariants as reports returned by :class:`pipls.PiPLSPathCV`.

    Parameters
    ----------
    n_components : int
        Component count represented by the report.
    predictor_rank : int
        Predictor rank represented by the report.
    n_splits : int
        Number of cross-validation splits.
    mean_test_score : float
        Mean configured test score.
    mean_response_standardized_mse : float
        Nonnegative mean response-standardized validation MSE.
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

    Notes
    -----
    ``estimate_kind="selection-conditioned"`` means the reported validation
    result was also used to choose ``n_components`` and/or ``predictor_rank``.
    It is therefore not an unbiased post-selection performance estimate. Arrays
    stored by the report are defensive, read-only copies.
    """

    n_components: int
    predictor_rank: int
    n_splits: int
    mean_test_score: float
    mean_response_standardized_mse: float
    estimate_kind: EstimateKind
    is_leave_one_out: bool
    oof_predictions: FloatArray | None = None
    oof_prediction_counts: IntArray | None = None
    pooled_oof_r2: float | None = None

    def __post_init__(self) -> None:
        n_components = _positive_int(self.n_components, name="n_components")
        predictor_rank = _positive_int(self.predictor_rank, name="predictor_rank")
        if n_components > predictor_rank:
            raise ValueError("n_components must not exceed predictor_rank.")
        n_splits = _positive_int(self.n_splits, name="n_splits")
        mean_test_score = _finite_float(self.mean_test_score, name="mean_test_score")
        mean_response_standardized_mse = _nonnegative_finite_float(
            self.mean_response_standardized_mse,
            name="mean_response_standardized_mse",
        )
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

        object.__setattr__(self, "n_components", n_components)
        object.__setattr__(self, "predictor_rank", predictor_rank)
        object.__setattr__(self, "n_splits", n_splits)
        object.__setattr__(self, "mean_test_score", mean_test_score)
        object.__setattr__(
            self,
            "mean_response_standardized_mse",
            mean_response_standardized_mse,
        )
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
                self.n_components,
                self.predictor_rank,
                self.n_splits,
                self.mean_test_score,
                self.mean_response_standardized_mse,
                self.estimate_kind,
                self.is_leave_one_out,
                self.oof_predictions,
                self.oof_prediction_counts,
                self.pooled_oof_r2,
            ),
        )

    @property
    def selection_conditioned(self) -> bool:
        """Whether the same CV result was used for parameter selection."""

        return self.estimate_kind == "selection-conditioned"

    @property
    def complete_oof_coverage(self) -> bool:
        """Whether every input row received at least one validation prediction."""

        counts = self.oof_prediction_counts
        return counts is not None and bool(np.all(counts > 0))
