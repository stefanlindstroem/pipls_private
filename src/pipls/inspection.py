"""Pure numerical utilities for fitted-model inspection."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.cross_decomposition import PLSRegression
from sklearn.utils.validation import check_is_fitted

from .decomposition import PiPLSDecomposition

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
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
    "PLSBiplotCoordinates",
    "PLSLatentStructure",
    "PLSObservationDiagnostics",
    "PiPLSDisplayFactors",
    "PredictionDiagnostics",
    "PredictionKind",
    "pipls_display_factors",
    "pls_biplot_coordinates",
    "pls_latent_structure",
    "pls_observation_diagnostics",
    "prediction_diagnostics",
]


@dataclass(frozen=True)
class PLSBiplotCoordinates:
    """Immutable balanced score-loading coordinates for two PLS components.

    ``sample_coordinates`` and ``predictor_coordinates`` preserve the selected
    score-loading reconstruction while giving both coordinate sets equal
    Euclidean norm within each component. Construct instances with
    :func:`pls_biplot_coordinates`.
    """

    sample_coordinates: FloatArray
    predictor_coordinates: FloatArray
    component_indices: IntArray
    scaling_factors: FloatArray

    @property
    def n_samples(self) -> int:
        """Number of represented observations."""

        return int(self.sample_coordinates.shape[0])

    @property
    def n_features(self) -> int:
        """Number of represented predictor variables."""

        return int(self.predictor_coordinates.shape[0])


@dataclass(frozen=True)
class PLSLatentStructure:
    """Immutable copies of public fitted ordinary-PLS quantities.

    ``coefficients`` follows the public scikit-learn ``coef_`` orientation and
    therefore has shape ``(n_targets, n_features)``. Construct instances with
    :func:`pls_latent_structure`.
    """

    x_scores: FloatArray
    x_loadings: FloatArray
    y_loadings: FloatArray
    coefficients: FloatArray

    @property
    def n_samples(self) -> int:
        """Number of observations represented by the X scores."""

        return int(self.x_scores.shape[0])

    @property
    def n_features(self) -> int:
        """Number of predictor variables."""

        return int(self.x_loadings.shape[0])

    @property
    def n_targets(self) -> int:
        """Number of response variables."""

        return int(self.y_loadings.shape[0])

    @property
    def n_components(self) -> int:
        """Number of retained ordinary-PLS components."""

        return int(self.x_scores.shape[1])


@dataclass(frozen=True)
class PLSObservationDiagnostics:
    """Immutable raw ordinary-PLS observation diagnostics.

    ``score_distance`` is the squared Mahalanobis distance of each supplied
    X score from the fitted training-score center, using the Moore--Penrose
    inverse of the fitted training-score covariance. ``x_reconstruction_residual``
    is the row-wise squared Euclidean residual after the public PLS
    transform/inverse-transform round trip.

    Construct instances with :func:`pls_observation_diagnostics`.
    """

    score_distance: FloatArray
    x_reconstruction_residual: FloatArray

    @property
    def n_samples(self) -> int:
        """Number of supplied observations."""

        return int(self.score_distance.shape[0])


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


def pls_biplot_coordinates(
    structure: PLSLatentStructure,
    *,
    components: Sequence[int] = (0, 1),
) -> PLSBiplotCoordinates:
    r"""Return balanced coordinates for a two-component PLS score-loading biplot.

    For selected score and X-loading columns $t_k$ and $p_k$, define

    .. math::

       a_k = \sqrt{\frac{\lVert p_k \rVert_2}{\lVert t_k \rVert_2}},
       \qquad \tilde t_k = a_k t_k,
       \qquad \tilde p_k = p_k / a_k.

    The transformed coordinates satisfy
    ``sample_coordinates @ predictor_coordinates.T == T_K @ P_K.T`` and
    have equal score and loading norm within each selected component.

    Parameters
    ----------
    structure:
        Immutable ordinary-PLS latent structure.
    components:
        Exactly two distinct zero-based component indices.

    Returns
    -------
    PLSBiplotCoordinates
        Read-only balanced sample and predictor coordinates.
    """

    if not isinstance(structure, PLSLatentStructure):
        raise TypeError("structure must be a PLSLatentStructure instance.")
    indices = _two_component_indices(components, size=structure.n_components)
    selected_scores = structure.x_scores[:, indices]
    selected_loadings = structure.x_loadings[:, indices]
    score_norms = np.linalg.norm(selected_scores, axis=0)
    loading_norms = np.linalg.norm(selected_loadings, axis=0)
    if np.any(score_norms == 0.0):
        raise ValueError("Selected PLS score columns must have nonzero norm.")
    if np.any(loading_norms == 0.0):
        raise ValueError("Selected PLS X-loading columns must have nonzero norm.")

    scaling_factors = np.sqrt(loading_norms / score_norms)
    sample_coordinates = selected_scores * scaling_factors[None, :]
    predictor_coordinates = selected_loadings / scaling_factors[None, :]

    return PLSBiplotCoordinates(
        sample_coordinates=_read_only(np.array(sample_coordinates, copy=True)),
        predictor_coordinates=_read_only(np.array(predictor_coordinates, copy=True)),
        component_indices=_read_only_ints(np.array(indices, dtype=np.int64)),
        scaling_factors=_read_only(np.array(scaling_factors, copy=True)),
    )


def pls_latent_structure(model: PLSRegression) -> PLSLatentStructure:
    """Return defensive copies of public fitted ``PLSRegression`` arrays.

    Parameters
    ----------
    model:
        A fitted :class:`sklearn.cross_decomposition.PLSRegression` estimator.

    Returns
    -------
    PLSLatentStructure
        Read-only X scores, X loadings, Y loadings, and regression
        coefficients.
    """

    if not isinstance(model, PLSRegression):
        raise TypeError("model must be a sklearn.cross_decomposition.PLSRegression.")
    check_is_fitted(
        model,
        attributes=["x_scores_", "x_loadings_", "y_loadings_", "coef_"],
    )

    x_scores = _finite_matrix(model.x_scores_, name="model.x_scores_")
    x_loadings = _finite_matrix(model.x_loadings_, name="model.x_loadings_")
    y_loadings = _finite_matrix(model.y_loadings_, name="model.y_loadings_")
    coefficients = _finite_matrix(model.coef_, name="model.coef_")

    n_components = x_scores.shape[1]
    if n_components == 0:
        raise ValueError("model must contain at least one retained component.")
    if x_loadings.shape[1] != n_components:
        raise ValueError(
            "model.x_scores_ and model.x_loadings_ must contain the same number of components."
        )
    if y_loadings.shape[1] != n_components:
        raise ValueError(
            "model.x_scores_ and model.y_loadings_ must contain the same number of components."
        )

    expected_coefficients = (y_loadings.shape[0], x_loadings.shape[0])
    if coefficients.shape != expected_coefficients:
        raise ValueError(
            "model.coef_ must have shape (n_targets, n_features): "
            f"expected {expected_coefficients}, got {coefficients.shape}."
        )

    return PLSLatentStructure(
        x_scores=_read_only(x_scores),
        x_loadings=_read_only(x_loadings),
        y_loadings=_read_only(y_loadings),
        coefficients=_read_only(coefficients),
    )


def pls_observation_diagnostics(
    model: PLSRegression,
    X: ArrayLike,
) -> PLSObservationDiagnostics:
    """Return raw score-distance and X-reconstruction diagnostics.

    The fitted training scores establish the score center and covariance.
    The supplied observations are transformed with the fitted model, and the
    covariance inverse is calculated with :func:`numpy.linalg.pinv` so that
    numerically rank-deficient score covariance remains well defined.

    No theoretical warning limits are calculated. The returned quantities are
    descriptive diagnostics whose interpretation depends on the fitted model
    and the scientific application.

    Parameters
    ----------
    model:
        A fitted :class:`sklearn.cross_decomposition.PLSRegression` estimator.
    X:
        Predictor observations with the fitted number of features.

    Returns
    -------
    PLSObservationDiagnostics
        Read-only raw score distances and squared X-reconstruction residuals.
    """

    if not isinstance(model, PLSRegression):
        raise TypeError("model must be a sklearn.cross_decomposition.PLSRegression.")
    check_is_fitted(model, attributes=["x_scores_", "x_loadings_"])

    training_scores = _finite_matrix(model.x_scores_, name="model.x_scores_")
    X_values = _finite_matrix(X, name="X")
    if training_scores.shape[0] < 2:
        raise ValueError("model.x_scores_ must contain at least two training observations.")
    if X_values.shape[1] != model.x_loadings_.shape[0]:
        raise ValueError(
            "X must contain the fitted number of predictor columns: "
            f"expected {model.x_loadings_.shape[0]}, got {X_values.shape[1]}."
        )

    score_center = np.mean(training_scores, axis=0)
    centered_training_scores = training_scores - score_center
    score_covariance = (
        centered_training_scores.T @ centered_training_scores
    ) / (training_scores.shape[0] - 1)
    inverse_covariance = np.linalg.pinv(score_covariance)

    transformed = _finite_matrix(model.transform(X), name="model.transform(X)")
    centered_scores = transformed - score_center
    score_distance = np.einsum(
        "ij,jk,ik->i",
        centered_scores,
        inverse_covariance,
        centered_scores,
    )
    reconstructed = _finite_matrix(
        model.inverse_transform(transformed),
        name="model.inverse_transform(model.transform(X))",
    )
    x_reconstruction_residual = np.sum((X_values - reconstructed) ** 2, axis=1)

    return PLSObservationDiagnostics(
        score_distance=_read_only(np.asarray(score_distance, dtype=np.float64)),
        x_reconstruction_residual=_read_only(
            np.asarray(x_reconstruction_residual, dtype=np.float64)
        ),
    )


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


def _two_component_indices(components: Sequence[int], *, size: int) -> tuple[int, int]:
    values = tuple(components)
    if len(values) != 2:
        raise ValueError("components must contain exactly two indices.")
    if any(isinstance(value, bool) or not isinstance(value, (int, np.integer)) for value in values):
        raise TypeError("components must contain integer indices.")
    first, second = (int(value) for value in values)
    if first == second:
        raise ValueError("components must contain two distinct indices.")
    if first < 0 or second < 0 or first >= size or second >= size:
        raise ValueError(
            f"components must lie in [0, {size - 1}]; got {(first, second)}."
        )
    return first, second


def _read_only(values: FloatArray) -> FloatArray:
    values.setflags(write=False)
    return values


def _read_only_ints(values: IntArray) -> IntArray:
    values.setflags(write=False)
    return values


def _read_only_signs(values: SignArray) -> SignArray:
    values.setflags(write=False)
    return values
