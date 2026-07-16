"""Theory-faithful fixed-parameter Pi-PLS numerical core.

This private module operates on predictor and response matrices that have already
been centered and, when requested by a caller, scaled. It contains no estimator,
cross-validation, or preprocessing policy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class PiPLSCoreResult:
    """Result of the fixed-parameter Pi-PLS construction.

    Attributes
    ----------
    Pi:
        Predictor basis with shape ``(p, r_pi)``.
    C:
        Response basis with shape ``(q, h)``.
    W:
        Least-squares map with shape ``(r_pi, h)``.
    P:
        Orthogonal predictor rotations with shape ``(p, h)``.
    D:
        Nonnegative diagonal dilation matrix with shape ``(h, h)``.
    Q:
        Orthogonal response rotations with shape ``(q, h)``.
    x_rank:
        Numerical rank of the supplied predictor matrix.
    rank_tolerance:
        Relative-scale tolerance used to determine ``x_rank``.
    """

    Pi: FloatArray
    C: FloatArray
    W: FloatArray
    P: FloatArray
    D: FloatArray
    Q: FloatArray
    x_rank: int
    rank_tolerance: float

    @property
    def regression_map(self) -> FloatArray:
        """Return the centered/scaled regression map with shape ``(p, q)``."""

        regression_map: FloatArray = self.P @ self.D @ self.Q.T
        return regression_map

    def predict(self, X: ArrayLike) -> FloatArray:
        """Predict centered/scaled responses from a compatible predictor matrix."""

        X_array = _as_finite_matrix(X, name="X")
        if X_array.shape[1] != self.P.shape[0]:
            raise ValueError(
                "X has an incompatible number of features: "
                f"expected {self.P.shape[0]}, got {X_array.shape[1]}."
            )
        return X_array @ self.regression_map


def fit_pipls_core(
    X: ArrayLike,
    Y: ArrayLike,
    *,
    predictor_rank: int,
    n_components: int,
) -> PiPLSCoreResult:
    r"""Fit the fixed-parameter Pi-PLS core to preprocessed matrices.

    Parameters
    ----------
    X:
        Centered, optionally scaled predictor matrix with shape ``(n, p)``.
    Y:
        Centered, optionally scaled response matrix with shape ``(n, q)``.
    predictor_rank:
        Predictor truncation rank $r_\pi$.
    n_components:
        Response-side latent dimension $h$.

    Returns
    -------
    PiPLSCoreResult
        The complete private factorization used by later estimator layers.

    Raises
    ------
    ValueError
        If arrays are nonfinite or incompatible, dimensions are inadmissible,
        or ``predictor_rank`` exceeds the numerical rank of ``X``.
    """

    X_array = _as_finite_matrix(X, name="X")
    Y_array = _as_finite_matrix(Y, name="Y")

    if X_array.shape[0] != Y_array.shape[0]:
        raise ValueError(
            "X and Y must contain the same number of samples: "
            f"got {X_array.shape[0]} and {Y_array.shape[0]}."
        )

    n_samples, n_features = X_array.shape
    n_targets = Y_array.shape[1]
    if n_samples == 0 or n_features == 0 or n_targets == 0:
        raise ValueError("X and Y must have at least one sample and one column.")

    r_pi = _as_positive_int(predictor_rank, name="predictor_rank")
    h = _as_positive_int(n_components, name="n_components")

    algebraic_limit = min(n_samples, n_features)
    if r_pi > algebraic_limit:
        raise ValueError(
            "predictor_rank must satisfy predictor_rank <= min(n_samples, n_features); "
            f"got predictor_rank={r_pi}, min(...)={algebraic_limit}."
        )
    if h > min(r_pi, n_targets):
        raise ValueError(
            "n_components must satisfy n_components <= min(predictor_rank, n_targets); "
            f"got n_components={h}, predictor_rank={r_pi}, n_targets={n_targets}."
        )

    _, x_singular_values, x_vt = np.linalg.svd(X_array, full_matrices=False)
    x_shape = (X_array.shape[0], X_array.shape[1])
    rank_tolerance = _svd_rank_tolerance(x_shape, x_singular_values)
    x_rank = int(np.count_nonzero(x_singular_values > rank_tolerance))
    if r_pi > x_rank:
        raise ValueError(
            "predictor_rank exceeds the numerical rank of X: "
            f"predictor_rank={r_pi}, numerical_rank={x_rank}, "
            f"tolerance={rank_tolerance:.6g}."
        )

    Pi = np.asarray(x_vt[:r_pi, :].T, dtype=np.float64)
    Z = X_array @ Pi

    cross_product = Z.T @ Y_array
    _, _, cross_vt = np.linalg.svd(cross_product, full_matrices=False)
    C = np.asarray(cross_vt[:h, :].T, dtype=np.float64)

    Y_C = Y_array @ C
    W_raw, _, _, _ = np.linalg.lstsq(Z, Y_C, rcond=None)
    W = cast(FloatArray, np.asarray(W_raw, dtype=np.float64))

    M, dilation, N_t = np.linalg.svd(W, full_matrices=False)
    M = np.asarray(M[:, :h], dtype=np.float64)
    dilation = np.asarray(dilation[:h], dtype=np.float64)
    N = np.asarray(N_t[:h, :].T, dtype=np.float64)

    P = np.asarray(Pi @ M, dtype=np.float64)
    D = np.diag(dilation).astype(np.float64, copy=False)
    Q = np.asarray(C @ N, dtype=np.float64)

    return PiPLSCoreResult(
        Pi=Pi,
        C=C,
        W=W,
        P=P,
        D=D,
        Q=Q,
        x_rank=x_rank,
        rank_tolerance=rank_tolerance,
    )


def _as_finite_matrix(value: ArrayLike, *, name: str) -> FloatArray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional array; got ndim={array.ndim}.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values.")
    return array


def _as_positive_int(value: int, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be a positive integer; got {value!r}.")
    integer = int(value)
    if integer < 1:
        raise ValueError(f"{name} must be at least 1; got {integer}.")
    return integer


def _svd_rank_tolerance(shape: tuple[int, int], singular_values: FloatArray) -> float:
    if singular_values.size == 0:
        return 0.0
    return float(max(shape) * np.finfo(np.float64).eps * singular_values[0])
