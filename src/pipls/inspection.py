"""Pure numerical utilities for fitted-model inspection."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal, Protocol, TypeAlias, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.utils.validation import check_is_fitted

from .decomposition import PiPLSDecomposition

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
PredictionKind: TypeAlias = Literal[
    "fitted values",
    "fixed-parameter OOF predictions",
    "selection-conditioned OOF predictions",
    "external test predictions",
]
"""Provenance label attached to prediction diagnostics."""


class _LatentStructureModel(Protocol):
    """Structural fitted-model contract used by shared PLS-family analysis."""

    x_scores_: ArrayLike
    x_loadings_: ArrayLike
    y_loadings_: ArrayLike
    coef_: ArrayLike

    def fit(self, X: ArrayLike, y: ArrayLike) -> Any:
        """Fit the estimator."""


class _ObservationModel(_LatentStructureModel, Protocol):
    """Latent-structure model with public X transform reconstruction methods."""

    def transform(self, X: ArrayLike) -> ArrayLike:
        """Transform predictor observations to X scores."""

    def inverse_transform(self, X: ArrayLike) -> ArrayLike:
        """Reconstruct predictor observations from X scores."""


_PREDICTION_KINDS: tuple[PredictionKind, ...] = (
    "fitted values",
    "fixed-parameter OOF predictions",
    "selection-conditioned OOF predictions",
    "external test predictions",
)

__all__ = [
    "BiplotCoordinates",
    "LatentStructure",
    "ObservationDiagnostics",
    "PiPLSDisplayFactors",
    "PredictionDiagnostics",
    "PredictionKind",
    "pipls_display_factors",
    "biplot_coordinates",
    "latent_structure",
    "observation_diagnostics",
    "prediction_diagnostics",
]


@dataclass(frozen=True)
class BiplotCoordinates:
    r"""Immutable balanced coordinates for a two-component PLS biplot.

    ``sample_coordinates`` and ``predictor_coordinates`` preserve the selected
    score-loading reconstruction while giving both coordinate sets equal
    Euclidean norm within each component. Construct instances with
    :func:`biplot_coordinates`.

    Attributes
    ----------
    sample_coordinates : ndarray of shape (n_samples, 2)
        Balanced X-score coordinates for the observations.
    predictor_coordinates : ndarray of shape (n_features, 2)
        Balanced X-loading coordinates for the predictors.
    component_indices : ndarray of shape (2,)
        Selected zero-based component indices.
    scaling_factors : ndarray of shape (2,)
        Positive factors applied to scores and inversely to loadings.
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
class LatentStructure:
    r"""Immutable copies of public fitted PLS-family quantities.

    The coefficient orientation follows ``PLSRegression`` and
    ``PiPLSRegression``. Construct instances with :func:`latent_structure`.

    Attributes
    ----------
    x_scores : ndarray of shape (n_samples, n_components)
        Fitted X-score matrix.
    x_loadings : ndarray of shape (n_features, n_components)
        X-loading matrix.
    y_loadings : ndarray of shape (n_targets, n_components)
        Y-loading matrix.
    coefficients : ndarray of shape (n_targets, n_features)
        Regression coefficients mapping predictors to responses.
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
        """Number of retained latent components."""

        return int(self.x_scores.shape[1])


@dataclass(frozen=True)
class ObservationDiagnostics:
    r"""Immutable raw PLS-family observation diagnostics.

    ``score_distance`` is the squared Mahalanobis distance from the fitted
    training-score center, using the Moore--Penrose inverse of the fitted
    training-score covariance. ``x_reconstruction_residual`` is the row-wise
    squared Euclidean residual after the public transform/inverse-transform round
    trip. No theoretical warning limits are attached.

    Attributes
    ----------
    score_distance : ndarray of shape (n_samples,)
        Raw squared score distances for the supplied observations.
    x_reconstruction_residual : ndarray of shape (n_samples,)
        Raw squared X-reconstruction residuals.
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

    The predictor and response direction columns use one deterministic display
    sign per component. Applying the same sign to both sides preserves the
    centered/scaled regression map $PDQ^{\mathsf T}$.

    Attributes
    ----------
    predictor_directions : ndarray of shape (n_features, n_components)
        Display-signed copy of $P$.
    dilation : ndarray of shape (n_components,)
        Diagonal values of $D$.
    response_directions : ndarray of shape (n_targets, n_components)
        Display-signed copy of $Q$.
    weighted_response_directions : ndarray of shape (n_targets, n_components)
        Columns $d_k q_{:k}$, equal to ``response_directions * dilation``.
    """

    predictor_directions: FloatArray
    dilation: FloatArray
    response_directions: FloatArray
    weighted_response_directions: FloatArray

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
    r"""Immutable standardized prediction and residual diagnostics.

    All response matrices are two-dimensional, including single-response input.
    Centers and sample standard deviations are estimated from ``observed`` and
    applied unchanged to ``predicted``.

    Attributes
    ----------
    observed, predicted, residual : ndarray of shape (n_samples, n_targets)
        Responses on their original scale, with residual defined as observed minus
        predicted.
    observed_standardized : ndarray of shape (n_samples, n_targets)
        Standardized observed responses.
    predicted_standardized : ndarray of shape (n_samples, n_targets)
        Standardized predicted responses.
    residual_standardized : ndarray of shape (n_samples, n_targets)
        Standardized residuals.
    response_centers, response_scales : ndarray of shape (n_targets,)
        Display centering and sample-standard-deviation vectors from the observed
        responses.
    standardized_rmse : ndarray of shape (n_targets,)
        Response-wise root mean squared standardized residual.
    prediction_kind : PredictionKind
        Explicit provenance of the supplied predictions.
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


def biplot_coordinates(
    structure: LatentStructure,
    *,
    components: Sequence[int] = (0, 1),
) -> BiplotCoordinates:
    r"""Return balanced coordinates for a two-component PLS score-loading biplot.

    For selected score and X-loading columns $t_k$ and $p_k$, define

    .. math::

       a_k = \sqrt{\frac{\lVert p_k \rVert_2}{\lVert t_k \rVert_2}},
       \qquad \tilde t_k = a_k t_k,
       \qquad \tilde p_k = p_k / a_k.

    The balanced coordinates preserve ``T_K @ P_K.T`` and have equal score and
    loading norm within each selected component.

    Parameters
    ----------
    structure : LatentStructure
        Extracted fitted PLS-family quantities.
    components : sequence of int, default=(0, 1)
        Exactly two distinct zero-based component indices.

    Returns
    -------
    BiplotCoordinates
        Read-only balanced sample and predictor coordinates.
    """

    if not isinstance(structure, LatentStructure):
        raise TypeError("structure must be a LatentStructure instance.")
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

    return BiplotCoordinates(
        sample_coordinates=_read_only(np.array(sample_coordinates, copy=True)),
        predictor_coordinates=_read_only(np.array(predictor_coordinates, copy=True)),
        component_indices=_read_only_ints(np.array(indices, dtype=np.int64)),
        scaling_factors=_read_only(np.array(scaling_factors, copy=True)),
    )


def latent_structure(model: object) -> LatentStructure:
    r"""Return defensive copies of public fitted PLS-family arrays.

    The function uses a structural fitted-model contract rather than a concrete
    estimator class. Compatible fitted models expose ``x_scores_``,
    ``x_loadings_``, ``y_loadings_``, and ``coef_`` with the usual PLS-family
    orientations.

    Parameters
    ----------
    model : object
        Fitted PLS-family estimator, such as :class:`pipls.PiPLSRegression` or
        scikit-learn ``PLSRegression``.

    Returns
    -------
    LatentStructure
        Read-only scores, loadings, and coefficients.
    """

    fitted = cast(
        _LatentStructureModel,
        _require_fitted_model(
            model,
            attributes=("x_scores_", "x_loadings_", "y_loadings_", "coef_"),
        ),
    )

    x_scores = _finite_matrix(fitted.x_scores_, name="model.x_scores_")
    x_loadings = _finite_matrix(fitted.x_loadings_, name="model.x_loadings_")
    y_loadings = _finite_matrix(fitted.y_loadings_, name="model.y_loadings_")
    coefficients = _finite_matrix(fitted.coef_, name="model.coef_")

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

    return LatentStructure(
        x_scores=_read_only(x_scores),
        x_loadings=_read_only(x_loadings),
        y_loadings=_read_only(y_loadings),
        coefficients=_read_only(coefficients),
    )


def observation_diagnostics(
    model: object,
    X: ArrayLike,
) -> ObservationDiagnostics:
    r"""Return raw score-distance and X-reconstruction diagnostics.

    The fitted training scores establish the score center and covariance. The
    supplied observations are transformed with the fitted model, and the covariance
    inverse is calculated with :func:`numpy.linalg.pinv`. No theoretical warning
    limits are calculated.

    Parameters
    ----------
    model : object
        Fitted PLS-family estimator exposing public X scores, X loadings,
        ``transform()``, and ``inverse_transform()``.
    X : array-like of shape (n_samples, n_features)
        Predictor observations on the fitted model's input scale.

    Returns
    -------
    ObservationDiagnostics
        Read-only raw score distances and squared X-reconstruction residuals.
    """

    fitted = cast(
        _ObservationModel,
        _require_fitted_model(
            model,
            attributes=("x_scores_", "x_loadings_"),
            methods=("transform", "inverse_transform"),
        ),
    )

    training_scores = _finite_matrix(fitted.x_scores_, name="model.x_scores_")
    x_loadings = _finite_matrix(fitted.x_loadings_, name="model.x_loadings_")
    X_values = _finite_matrix(X, name="X")
    if training_scores.shape[0] < 2:
        raise ValueError("model.x_scores_ must contain at least two training observations.")
    if X_values.shape[1] != x_loadings.shape[0]:
        raise ValueError(
            "X must contain the fitted number of predictor columns: "
            f"expected {x_loadings.shape[0]}, got {X_values.shape[1]}."
        )

    score_center = np.mean(training_scores, axis=0)
    centered_training_scores = training_scores - score_center
    score_covariance = (centered_training_scores.T @ centered_training_scores) / (
        training_scores.shape[0] - 1
    )
    inverse_covariance = np.linalg.pinv(score_covariance)

    transformed = _finite_matrix(fitted.transform(X), name="model.transform(X)")
    centered_scores = transformed - score_center
    score_distance = np.einsum(
        "ij,jk,ik->i",
        centered_scores,
        inverse_covariance,
        centered_scores,
    )
    reconstructed = _finite_matrix(
        fitted.inverse_transform(transformed),
        name="model.inverse_transform(model.transform(X))",
    )
    x_reconstruction_residual = np.sum((X_values - reconstructed) ** 2, axis=1)

    return ObservationDiagnostics(
        score_distance=_read_only(np.asarray(score_distance, dtype=np.float64)),
        x_reconstruction_residual=_read_only(
            np.asarray(x_reconstruction_residual, dtype=np.float64)
        ),
    )


def pipls_display_factors(decomposition: PiPLSDecomposition) -> PiPLSDisplayFactors:
    r"""Return copied Pi-PLS factors with deterministic display signs.

    For each component, the first largest-magnitude entry of the predictor
    direction is made nonnegative. The same sign is applied to the corresponding
    response direction, preserving $PDQ^{\mathsf T}$.

    Parameters
    ----------
    decomposition : pipls.PiPLSDecomposition
        Public fitted Pi-PLS decomposition.

    Returns
    -------
    PiPLSDisplayFactors
        Read-only display factors and the applied component signs.
    """

    if not isinstance(decomposition, PiPLSDecomposition):
        raise TypeError("decomposition must be a PiPLSDecomposition.")

    predictor_directions = _finite_matrix(
        decomposition.predictor_rotations,
        name="decomposition.predictor_rotations",
    )
    response_directions = _finite_matrix(
        decomposition.response_rotations,
        name="decomposition.response_rotations",
    )
    dilation = _finite_vector(decomposition.dilation, name="decomposition.dilation")

    n_components = predictor_directions.shape[1]
    if n_components == 0:
        raise ValueError("decomposition must contain at least one component.")
    if response_directions.shape[1] != n_components:
        raise ValueError(
            "decomposition predictor and response rotations must contain "
            "the same number of components."
        )
    if dilation.shape != (n_components,):
        raise ValueError(
            "decomposition.dilation must contain one value per component: "
            f"expected {(n_components,)}, got {dilation.shape}."
        )
    if np.any(dilation < 0.0):
        raise ValueError("decomposition.dilation must contain nonnegative values.")

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
    )


def prediction_diagnostics(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    *,
    prediction_kind: PredictionKind,
) -> PredictionDiagnostics:
    r"""Return standardized observed, predicted, and residual diagnostics.

    Residuals use observed minus predicted. Response centers and scales are
    calculated from the observed responses. Scales are sample standard deviations
    with ``ddof=1`` and are applied unchanged to predictions.

    Parameters
    ----------
    y_true, y_pred : array-like of shape (n_samples,) or (n_samples, n_targets)
        Aligned observed and predicted responses.
    prediction_kind : PredictionKind
        Explicit provenance label for the predictions.

    Returns
    -------
    PredictionDiagnostics
        Read-only response matrices, standardization statistics, and response-wise
        standardized RMSE.
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


def _require_fitted_model(
    model: object,
    *,
    attributes: Sequence[str],
    methods: Sequence[str] = (),
) -> object:
    """Validate the structural contract and return the fitted model."""

    if not callable(getattr(model, "fit", None)):
        raise TypeError("model must be a PLS-family estimator with a callable fit method.")
    missing_methods = [name for name in methods if not callable(getattr(model, name, None))]
    if missing_methods:
        joined = ", ".join(f"{name}()" for name in missing_methods)
        raise TypeError(f"model must expose callable {joined} methods.")
    check_is_fitted(cast(Any, model), attributes=list(attributes))
    return model


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
        raise ValueError(f"components must lie in [0, {size - 1}]; got {(first, second)}.")
    return first, second


def _read_only(values: FloatArray) -> FloatArray:
    values.setflags(write=False)
    return values


def _read_only_ints(values: IntArray) -> IntArray:
    values.setflags(write=False)
    return values
