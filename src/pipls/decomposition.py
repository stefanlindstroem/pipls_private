"""Public immutable representation of a fitted Pi-PLS factorization."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ._core import PiPLSCoreResult, ResolvedSVDSolver

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class PiPLSDecomposition:
    r"""Pi-PLS factorization and numerical diagnostics.

    The matrices follow the notation used by the Pi-PLS method. The public
    arrays are read-only copies so that the fitted estimator cannot be changed
    accidentally through this result object.

    Attributes
    ----------
    Pi:
        Truncated predictor basis with shape ``(n_features, predictor_rank)``.
    C:
        Response basis with shape ``(n_targets, n_components)``.
    W:
        Least-squares coupling map with shape
        ``(predictor_rank, n_components)``.
    P, D, Q:
        Orthogonal-diagonal-orthogonal factorization of the centered/scaled
        regression map, so that ``P @ D @ Q.T`` is that map.
    dilation:
        Diagonal of ``D``.
    x_rank:
        Complete numerical predictor rank under full SVD, or a verified lower
        bound under truncated randomized SVD.
    x_rank_is_exact:
        Whether ``x_rank`` is the complete numerical rank.
    rank_tolerance:
        Tolerance used to classify retained predictor singular values.
    predictor_svd_solver:
        Predictor SVD implementation actually used.
    """

    Pi: FloatArray
    C: FloatArray
    W: FloatArray
    P: FloatArray
    D: FloatArray
    Q: FloatArray
    dilation: FloatArray
    x_rank: int
    x_rank_is_exact: bool
    rank_tolerance: float
    predictor_svd_solver: ResolvedSVDSolver

    @classmethod
    def _from_core_result(cls, result: PiPLSCoreResult) -> PiPLSDecomposition:
        return cls(
            Pi=_read_only_copy(result.Pi),
            C=_read_only_copy(result.C),
            W=_read_only_copy(result.W),
            P=_read_only_copy(result.P),
            D=_read_only_copy(result.D),
            Q=_read_only_copy(result.Q),
            dilation=_read_only_copy(np.diag(result.D)),
            x_rank=result.x_rank,
            x_rank_is_exact=result.x_rank_is_exact,
            rank_tolerance=result.rank_tolerance,
            predictor_svd_solver=result.predictor_svd_solver,
        )

    @property
    def regression_map(self) -> FloatArray:
        """Return the centered/scaled regression map ``P @ D @ Q.T``."""

        return np.asarray(self.P @ self.D @ self.Q.T, dtype=np.float64)

    @property
    def predictor_basis(self) -> FloatArray:
        """Descriptive alias for :attr:`Pi`."""

        return self.Pi

    @property
    def response_basis(self) -> FloatArray:
        """Descriptive alias for :attr:`C`."""

        return self.C

    @property
    def least_squares_map(self) -> FloatArray:
        """Descriptive alias for :attr:`W`."""

        return self.W

    @property
    def predictor_rotations(self) -> FloatArray:
        """Descriptive alias for :attr:`P`."""

        return self.P

    @property
    def dilation_matrix(self) -> FloatArray:
        """Descriptive alias for :attr:`D`."""

        return self.D

    @property
    def response_rotations(self) -> FloatArray:
        """Descriptive alias for :attr:`Q`."""

        return self.Q


def _read_only_copy(value: FloatArray) -> FloatArray:
    array = np.array(value, dtype=np.float64, copy=True)
    array.setflags(write=False)
    return array
