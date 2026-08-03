"""Structured out-of-fold reporting for public Pi-PLS estimators."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._result_validation import (
    _boolean,
    _finite_float,
    _read_only_float_array,
    _read_only_int_array,
)
from .component_path import PiPLSComponentResult

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.intp]


def _validated_oof_fields(
    *,
    is_leave_one_out: object,
    oof_predictions: object,
    oof_prediction_counts: object,
    pooled_oof_r2: object,
) -> tuple[bool, FloatArray | None, IntArray | None, float | None]:
    """Normalize common immutable OOF report fields."""

    leave_one_out = _boolean(is_leave_one_out, name="is_leave_one_out")
    pooled = (
        None
        if pooled_oof_r2 is None
        else _finite_float(pooled_oof_r2, name="pooled_oof_r2")
    )

    predictions: FloatArray | None = None
    counts: IntArray | None = None
    if oof_predictions is None:
        if oof_prediction_counts is not None or pooled is not None:
            raise ValueError("OOF counts and pooled OOF R2 require oof_predictions.")
        return leave_one_out, predictions, counts, pooled

    if oof_prediction_counts is None:
        raise ValueError("oof_prediction_counts are required with oof_predictions.")
    predictions = _read_only_float_array(
        cast(ArrayLike, oof_predictions),
        name="oof_predictions",
        ndim=np.asarray(oof_predictions).ndim,
        require_finite=False,
    )
    if predictions.ndim not in (1, 2):
        raise ValueError("oof_predictions must be one- or two-dimensional.")
    if predictions.shape[0] == 0:
        raise ValueError("oof_predictions must contain at least one row.")
    counts = _read_only_int_array(
        cast(ArrayLike, oof_prediction_counts),
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
    if pooled is not None and int(np.count_nonzero(covered)) < 2:
        raise ValueError("pooled_oof_r2 requires at least two rows with OOF coverage.")
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

    return leave_one_out, predictions, counts, pooled


@dataclass(frozen=True)
class PiPLSOOFReport:
    """Immutable OOF diagnostics for one existing Pi-PLS selection.

    Parameters
    ----------
    selection : PiPLSComponentResult
        Exact immutable selection evaluated by the report.
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
    A report returned by :meth:`pipls.PiPLSSearchCV.oof_report` reuses the search
    splits that produced the supplied selection. It is therefore a
    selection-conditioned diagnostic rather than an unbiased post-selection
    performance estimate. Arrays are defensive, read-only copies.
    """

    selection: PiPLSComponentResult
    is_leave_one_out: bool
    oof_predictions: FloatArray | None = None
    oof_prediction_counts: IntArray | None = None
    pooled_oof_r2: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.selection, PiPLSComponentResult):
            raise TypeError("selection must be a PiPLSComponentResult.")
        leave_one_out, predictions, counts, pooled = _validated_oof_fields(
            is_leave_one_out=self.is_leave_one_out,
            oof_predictions=self.oof_predictions,
            oof_prediction_counts=self.oof_prediction_counts,
            pooled_oof_r2=self.pooled_oof_r2,
        )
        object.__setattr__(self, "is_leave_one_out", leave_one_out)
        object.__setattr__(self, "oof_predictions", predictions)
        object.__setattr__(self, "oof_prediction_counts", counts)
        object.__setattr__(self, "pooled_oof_r2", pooled)

    def __reduce__(self) -> tuple[type[PiPLSOOFReport], tuple[object, ...]]:
        """Reconstruct through validation so unpickled arrays remain read-only."""

        return (
            type(self),
            (
                self.selection,
                self.is_leave_one_out,
                self.oof_predictions,
                self.oof_prediction_counts,
                self.pooled_oof_r2,
            ),
        )

    @property
    def n_components(self) -> int:
        """Number of paired latent modes represented by the report."""

        return self.selection.n_components

    @property
    def predictor_rank(self) -> int:
        """Predictor rank represented by the report."""

        return self.selection.predictor_rank

    @property
    def n_splits(self) -> int:
        """Number of cross-validation splits."""

        return self.selection.n_splits

    @property
    def mean_test_score(self) -> float:
        """Mean configured test score."""

        return self.selection.mean_test_score

    @property
    def cv_mse_mean(self) -> float:
        """Mean response-standardized validation MSE."""

        return self.selection.cv_mse_mean

    @property
    def has_complete_oof_coverage(self) -> bool:
        """Whether every input row received at least one validation prediction."""

        counts = self.oof_prediction_counts
        return counts is not None and bool(np.all(counts > 0))
