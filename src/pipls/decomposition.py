"""Public immutable representation of a fitted Pi-PLS factorization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import numpy as np
from numpy.typing import NDArray

from ._core import PiPLSCoreResult, ResolvedSVDSolver
from ._result_validation import (
    _boolean,
    _literal_string,
    _nonnegative_finite_float,
    _positive_int,
    _read_only_float_array,
)

FloatArray = NDArray[np.float64]
_ALLOWED_RESOLVED_SVD_SOLVERS = frozenset({"full", "randomized"})


@dataclass(frozen=True)
class PiPLSDecomposition:
    r"""Interpretable Pi-PLS factorization and numerical diagnostics.

    Instances are returned through :attr:`pipls.PiPLSRegression.decomposition_`.
    Direct construction validates the same shape, scalar, and immutability
    invariants as estimator-produced instances. Internal construction matrices
    used before the final factorization are intentionally not exposed.

    Attributes
    ----------
    predictor_rotations : ndarray of shape (n_features, n_components)
        Orthonormal predictor directions $P$ of the centered/scaled regression
        map. The field name is retained for API compatibility.
    dilation : ndarray of shape (n_components,)
        Nonnegative dilations $d_k=D_{kk}$ of the paired latent modes.
    response_rotations : ndarray of shape (n_targets, n_components)
        Orthonormal response directions $Q$. The field name is retained for API
        compatibility.
    predictor_numerical_rank : int
        Complete numerical predictor rank under full SVD, or a verified lower
        bound under truncated randomized SVD.
    predictor_numerical_rank_is_exact : bool
        Whether ``predictor_numerical_rank`` is the complete numerical rank.
    rank_tolerance : float
        Nonnegative tolerance used to classify retained predictor singular values.
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

    def __post_init__(self) -> None:
        predictor_rotations = _read_only_float_array(
            self.predictor_rotations,
            name="predictor_rotations",
            ndim=2,
        )
        dilation = _read_only_float_array(self.dilation, name="dilation")
        response_rotations = _read_only_float_array(
            self.response_rotations,
            name="response_rotations",
            ndim=2,
        )
        n_components = int(dilation.size)
        if n_components == 0:
            raise ValueError("A decomposition must contain at least one component.")
        if predictor_rotations.shape[1] != n_components:
            raise ValueError(
                "predictor_rotations and dilation must contain the same number "
                "of components."
            )
        if response_rotations.shape[1] != n_components:
            raise ValueError(
                "response_rotations and dilation must contain the same number "
                "of components."
            )
        if predictor_rotations.shape[0] == 0 or response_rotations.shape[0] == 0:
            raise ValueError("Rotation arrays must contain at least one row.")
        if np.any(dilation < 0.0):
            raise ValueError("dilation must contain nonnegative values.")

        predictor_numerical_rank = _positive_int(
            self.predictor_numerical_rank,
            name="predictor_numerical_rank",
        )
        if predictor_numerical_rank < n_components:
            raise ValueError(
                "predictor_numerical_rank must not be smaller than the number "
                "of components."
            )
        predictor_numerical_rank_is_exact = _boolean(
            self.predictor_numerical_rank_is_exact,
            name="predictor_numerical_rank_is_exact",
        )
        rank_tolerance = _nonnegative_finite_float(
            self.rank_tolerance,
            name="rank_tolerance",
        )
        predictor_svd_solver = cast(
            ResolvedSVDSolver,
            _literal_string(
                self.predictor_svd_solver,
                name="predictor_svd_solver",
                allowed=_ALLOWED_RESOLVED_SVD_SOLVERS,
            ),
        )
        if predictor_svd_solver == "full" and not predictor_numerical_rank_is_exact:
            raise ValueError(
                'predictor_numerical_rank_is_exact must be true when '
                'predictor_svd_solver="full".'
            )
        if predictor_svd_solver == "randomized" and predictor_numerical_rank_is_exact:
            raise ValueError(
                'predictor_numerical_rank_is_exact must be false when '
                'predictor_svd_solver="randomized".'
            )

        object.__setattr__(self, "predictor_rotations", predictor_rotations)
        object.__setattr__(self, "dilation", dilation)
        object.__setattr__(self, "response_rotations", response_rotations)
        object.__setattr__(self, "predictor_numerical_rank", predictor_numerical_rank)
        object.__setattr__(
            self,
            "predictor_numerical_rank_is_exact",
            predictor_numerical_rank_is_exact,
        )
        object.__setattr__(self, "rank_tolerance", rank_tolerance)
        object.__setattr__(self, "predictor_svd_solver", predictor_svd_solver)

    def __reduce__(self) -> tuple[type[PiPLSDecomposition], tuple[object, ...]]:
        """Reconstruct through validation so unpickled arrays remain read-only."""

        return (
            type(self),
            (
                self.predictor_rotations,
                self.dilation,
                self.response_rotations,
                self.predictor_numerical_rank,
                self.predictor_numerical_rank_is_exact,
                self.rank_tolerance,
                self.predictor_svd_solver,
            ),
        )

    @classmethod
    def _from_core_result(cls, result: PiPLSCoreResult) -> PiPLSDecomposition:
        return cls(
            predictor_rotations=result.P,
            dilation=np.diag(result.D),
            response_rotations=result.Q,
            predictor_numerical_rank=result.x_rank,
            predictor_numerical_rank_is_exact=result.x_rank_is_exact,
            rank_tolerance=result.rank_tolerance,
            predictor_svd_solver=result.predictor_svd_solver,
        )

    @property
    def standardized_regression_map(self) -> FloatArray:
        r"""Return the centered/scaled regression map $P D Q^{\mathsf T}$."""

        return _read_only_float_array(
            np.asarray(
                (self.predictor_rotations * self.dilation[None, :])
                @ self.response_rotations.T,
                dtype=np.float64,
            ),
            name="standardized_regression_map",
            ndim=2,
        )
