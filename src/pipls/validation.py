"""Structured cross-validation reporting for public Pi-PLS estimators."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.intp]
EstimateKind = Literal["selection-conditioned", "fixed-parameter"]


@dataclass(frozen=True)
class PiPLSValidationReport:
    """Immutable summary of the CV result attached to a fitted Pi-PLS object.

    ``estimate_kind="selection-conditioned"`` means the reported validation
    result was also used to choose ``n_components`` and/or ``predictor_rank``.
    It is therefore not an unbiased post-selection performance estimate.
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
        if self.oof_predictions is None:
            if self.oof_prediction_counts is not None or self.pooled_oof_r2 is not None:
                raise ValueError(
                    "OOF counts and pooled OOF R2 require oof_predictions."
                )
            return
        if self.oof_prediction_counts is None:
            raise ValueError("oof_prediction_counts are required with oof_predictions.")

        predictions = np.array(self.oof_predictions, dtype=np.float64, copy=True)
        counts = np.array(self.oof_prediction_counts, dtype=np.intp, copy=True)
        if predictions.ndim not in (1, 2):
            raise ValueError("oof_predictions must be one- or two-dimensional.")
        if counts.ndim != 1 or counts.shape[0] != predictions.shape[0]:
            raise ValueError(
                "oof_prediction_counts must contain one value per prediction row."
            )
        predictions.setflags(write=False)
        counts.setflags(write=False)
        object.__setattr__(self, "oof_predictions", predictions)
        object.__setattr__(self, "oof_prediction_counts", counts)

    @property
    def selection_conditioned(self) -> bool:
        """Whether the same CV result was used for parameter selection."""

        return self.estimate_kind == "selection-conditioned"

    @property
    def complete_oof_coverage(self) -> bool:
        """Whether every input row received at least one validation prediction."""

        counts = self.oof_prediction_counts
        return counts is not None and bool(np.all(counts > 0))
