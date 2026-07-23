"""Public immutable representation of a fitted Pi-PLS factorization."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ._core import PiPLSCoreResult, ResolvedSVDSolver

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class PiPLSDecomposition:
    r"""Interpretable Pi-PLS factorization and numerical diagnostics.

    Instances are returned through :attr:`pipls.PiPLSRegression.decomposition_`.
    The public arrays are read-only copies so that the fitted estimator cannot
    be changed accidentally through this result object. Internal construction
    matrices used before the final factorization are intentionally not exposed.

    Attributes
    ----------
    predictor_rotations : ndarray of shape (n_features, n_components)
        Orthogonal predictor rotations $P$ of the centered/scaled regression
        map.
    dilation : ndarray of shape (n_components,)
        Nonnegative diagonal values of $D$.
    response_rotations : ndarray of shape (n_targets, n_components)
        Orthogonal response rotations $Q$.
    predictor_numerical_rank : int
        Complete numerical predictor rank under full SVD, or a verified lower
        bound under truncated randomized SVD.
    predictor_numerical_rank_is_exact : bool
        Whether ``predictor_numerical_rank`` is the complete numerical rank.
    rank_tolerance : float
        Tolerance used to classify retained predictor singular values.
    predictor_svd_solver : {"full", "randomized"}
        Predictor SVD implementation actually used.
    """

    predictor_rotations: FloatArray
    dilation: FloatArray
    response_rotations: FloatArray
    predictor_numerical_rank: int
    predictor_numerical_rank_is_exact: bool
    rank_tolerance: float
    predictor_svd_solver: ResolvedSVDSolver

    @classmethod
    def _from_core_result(cls, result: PiPLSCoreResult) -> PiPLSDecomposition:
        return cls(
            predictor_rotations=_read_only_copy(result.P),
            dilation=_read_only_copy(np.diag(result.D)),
            response_rotations=_read_only_copy(result.Q),
            predictor_numerical_rank=result.x_rank,
            predictor_numerical_rank_is_exact=result.x_rank_is_exact,
            rank_tolerance=result.rank_tolerance,
            predictor_svd_solver=result.predictor_svd_solver,
        )

    @property
    def standardized_regression_map(self) -> FloatArray:
        r"""Return the centered/scaled regression map $P D Q^{\mathsf T}$."""

        return _read_only_copy(
            np.asarray(
                (self.predictor_rotations * self.dilation[None, :])
                @ self.response_rotations.T,
                dtype=np.float64,
            )
        )


def _read_only_copy(value: FloatArray) -> FloatArray:
    array = np.array(value, dtype=np.float64, copy=True)
    array.setflags(write=False)
    return array
