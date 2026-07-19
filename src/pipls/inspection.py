"""Pure numerical utilities for fitted-model inspection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .decomposition import PiPLSDecomposition

FloatArray = NDArray[np.float64]
SignArray = NDArray[np.int8]
PredictionKind: TypeAlias = Literal[
    "fitted values",
    "fixed-parameter OOF predictions",
    "selection-conditioned OOF predictions",
    "external test predictions",
]

_PREDICTION_KINDS: tuple[PredictionKind, ...] = (
    "fitted values",
    "fixed-parameter OOF predictions",
    "selection-conditioned OOF predictions",
    "external test predictions",
)

__all__ = [
    "PiPLSDisplayFactors",
    "PredictionDiagnostics",
    "PredictionKind",
    "pipls_display_factors",
    "prediction_diagnostics",
]


@dataclass(frozen=True)
class PiPLSDisplayFactors:
    r"""Immutable display-oriented copy of a Pi-PLS factorization.

    The columns of ``predictor_directions`` and ``response_directions`` use a
    deterministic display sign. Applying the same sign to both sides preserves
    the centered/scaled regression map. ``weighted_response_directions`` is
    ``response_directions * dilation[None, :]``.

    Construct instances with :func:`pipls_display_factors`.
    """

    predictor_directions: FloatArray
    dilation: FloatArray
    response_directions: FloatArray
    weighted_response_directions: FloatArray
    component_signs: SignArray

    @property
    def n_features(self) -> int:
        """Number of predictor variables represented by the factors."""

        return int(self.predictor_directions.shape[0])

    @property
    def n_targets(self) -> int:
        """Number of response variables represented by the factors."""

        return int(self.response_directions.shape[0])

    @property
    def n_components(self) -> int:
        """Number of retained Pi-PLS components."""

        return int(self.dilation.shape[0])


@dataclass(frozen=True)
class PredictionDiagnostics:
    """Immutable standardized prediction and residual diagnostics.

    All response arrays have shape ``(n_samples, n_targets)`` even when the
    inputs were one-dimensional. Centers and sample standard deviations are
    estimated from ``observed`` and then applied unchanged to ``predicted``.

    Construct instances with :func:`prediction_diagnostics`.
    """

    observed: FloatArray
    predicted: FloatArray
    residual: FloatArray
    observed_standardized: FloatArray
    predicted_standardized: FloatArray
    residual_standardized: FloatArray
    response_centers: FloatArray
    response_scales: FloatArray
    standardized_rmse: FloatArray
    prediction_kind: PredictionKind

    @property
    def n_samples(self) -> int:
        """Number of aligned observations."""

        return int(self.observed.shape[0])

    @property
    def n_targets(self) -> int:
        """Number of response variables."""

        return int(self.observed.shape[1])


def pipls_display_factors(decomposition: PiPLSDecomposition) -> PiPLSDisplayFactors:
    r"""Return copied Pi-PLS factors with deterministic display signs.

    For each component, the first largest-magnitude entry of the predictor
    direction is made nonnegative. The same sign is applied to the corresponding
    response direction, preserving ``P @ D @ Q.T``. A zero predictor direction
    is left unchanged.

    Parameters
    ----------
    decomposition:
        Public fitted Pi-PLS decomposition.

    Returns
    -------
    PiPLSDisplayFactors
        Read-only copies of the display factors and applied component signs.
    """

    if not isinstance(decomposition, PiPLSDecomposition):
        raise TypeError("decomposition must be a PiPLSDecomposition.")

    predictor_directions = _finite_matrix(decomposition.P, name="decomposition.P")
    response_directions = _finite_matrix(decomposition.Q, name="decomposition.Q")
    dilation = _finite_vector(decomposition.dilation, name="decomposition.dilation")
    dilation_matrix = _finite_matrix(decomposition.D, name="decomposition.D")

    n_components = predictor_directions.shape[1]
    if n_components == 0:
        raise ValueError("decomposition must contain at least one component.")
    if response_directions.shape[1] != n_components:
        raise ValueError(
            "decomposition.P and decomposition.Q must contain the same number of components."
        )
    if dilation.shape != (n_components,):
        raise ValueError(
            "decomposition.dilation must contain one value per component: "
            f"expected {(n_components,)}, got {dilation.shape}."
        )
    if dilation_matrix.shape != (n_components, n_components):
        raise ValueError(
            "decomposition.D must be square with one row and column per component: "
            f"expected {(n_components, n_components)}, got {dilation_matrix.shape}."
        )
    if np.any(dilation < 0.0):
        raise ValueError("decomposition.dilation must contain nonnegative values.")
    if not np.array_equal(dilation_matrix, np.diag(dilation)):
        raise ValueError("decomposition.D must equal diag(decomposition.dilation).")

    component_signs = np.ones(n_components, dtype=np.int8)
    for component in range(n_components):
        column = predictor_directions[:, component]
        pivot = int(np.argmax(np.abs(column)))
        if column[pivot] < 0.0:
            component_signs[component] = -1

    predictor_directions *= component_signs[None, :]
    response_directions *= component_signs[None, :]
    weighted_response_directions = response_directions * dilation[None, :]

    return PiPLSDisplayFactors(
        predictor_directions=_read_only(predictor_directions),
        dilation=_read_only(dilation),
        response_directions=_read_only(response_directions),
        weighted_response_directions=_read_only(weighted_response_directions),
        component_signs=_read_only_signs(component_signs),
    )


def prediction_diagnostics(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    *,
    prediction_kind: PredictionKind,
) -> PredictionDiagnostics:
    r"""Return standardized observed, predicted, and residual diagnostics.

    Residuals use ``observed - predicted``. Response centers and scales are
    calculated from the observed responses. Scales are sample standard
    deviations using ``ddof=1`` and are then applied unchanged to predictions.

    Parameters
    ----------
    y_true, y_pred:
        Aligned observed and predicted responses. One- and two-dimensional
        inputs are accepted.
    prediction_kind:
        Explicit provenance label for the predictions.

    Returns
    -------
    PredictionDiagnostics
        Read-only response matrices, display standardization statistics, and
        response-wise standardized RMSE.
    """

    if prediction_kind not in _PREDICTION_KINDS:
        allowed = ", ".join(repr(value) for value in _PREDICTION_KINDS)
        raise ValueError(f"prediction_kind must be one of {allowed}.")

    observed = _response_matrix(y_true, name="y_true")
    predicted = _response_matrix(y_pred, name="y_pred")
    if observed.shape != predicted.shape:
        raise ValueError(
            "y_true and y_pred must have identical shapes after response normalization: "
            f"got {observed.shape} and {predicted.shape}."
        )
    if observed.shape[0] < 2:
        raise ValueError("Prediction diagnostics require at least two observations.")

    response_centers = np.mean(observed, axis=0)
    response_scales = np.std(observed, axis=0, ddof=1)
    constant = np.flatnonzero(response_scales == 0.0)
    if constant.size:
        columns = ", ".join(str(int(index)) for index in constant)
        raise ValueError(f"y_true contains constant response columns at indices: {columns}.")

    residual = observed - predicted
    observed_standardized = (observed - response_centers[None, :]) / response_scales[None, :]
    predicted_standardized = (predicted - response_centers[None, :]) / response_scales[None, :]
    residual_standardized = residual / response_scales[None, :]
    standardized_rmse = np.sqrt(np.mean(np.square(residual_standardized), axis=0))

    return PredictionDiagnostics(
        observed=_read_only(observed),
        predicted=_read_only(predicted),
        residual=_read_only(residual),
        observed_standardized=_read_only(observed_standardized),
        predicted_standardized=_read_only(predicted_standardized),
        residual_standardized=_read_only(residual_standardized),
        response_centers=_read_only(response_centers),
        response_scales=_read_only(response_scales),
        standardized_rmse=_read_only(standardized_rmse),
        prediction_kind=prediction_kind,
    )


def _response_matrix(values: ArrayLike, *, name: str) -> FloatArray:
    array = np.array(values, dtype=np.float64, copy=True)
    if array.ndim == 1:
        array = array.reshape(-1, 1)
    if array.ndim != 2:
        raise ValueError(f"{name} must be one- or two-dimensional; got shape {array.shape}.")
    if array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError(f"{name} must contain at least one sample and one response.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values.")
    return array


def _finite_matrix(values: ArrayLike, *, name: str) -> FloatArray:
    array = np.array(values, dtype=np.float64, copy=True)
    if array.ndim != 2:
        raise ValueError(f"{name} must be two-dimensional; got shape {array.shape}.")
    if array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError(f"{name} must contain at least one row and one column.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values.")
    return array


def _finite_vector(values: ArrayLike, *, name: str) -> FloatArray:
    array = np.array(values, dtype=np.float64, copy=True)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional; got shape {array.shape}.")
    if array.shape[0] == 0:
        raise ValueError(f"{name} must contain at least one value.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values.")
    return array


def _read_only(values: FloatArray) -> FloatArray:
    values.setflags(write=False)
    return values


def _read_only_signs(values: SignArray) -> SignArray:
    values.setflags(write=False)
    return values
