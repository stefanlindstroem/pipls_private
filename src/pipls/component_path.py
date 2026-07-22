"""Immutable concise results for a fitted Pi-PLS component path."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.intp]
StringArray = NDArray[np.str_]
PredictorRankPolicy = Literal["optimized", "fixed", "maximum"]
_ALLOWED_PREDICTOR_RANK_POLICIES = frozenset({"optimized", "fixed", "maximum"})


@dataclass(frozen=True)
class PiPLSComponentResult:
    """Conditionally selected result for one component count.

    Attributes
    ----------
    n_components : int
        Evaluated component count.
    predictor_rank : int
        Predictor rank selected conditionally for ``n_components``.
    predictor_rank_policy : {"optimized", "fixed", "maximum"}
        Interpretation of the predictor-rank specification used by the path.
    mean_test_score : float
        Mean configured test score for the selected candidate.
    cv_mse_mean : float
        Mean response-standardized validation MSE.
    cv_mse_fold_sd : float
        Population standard deviation of response-standardized MSE across folds.
    n_splits : int
        Number of cross-validation splits.
    """

    n_components: int
    predictor_rank: int
    predictor_rank_policy: PredictorRankPolicy
    mean_test_score: float
    cv_mse_mean: float
    cv_mse_fold_sd: float
    n_splits: int


@dataclass(frozen=True)
class PiPLSPredictorRankProfile:
    """Immutable evaluated predictor-rank results for one component count.

    Arrays contain the predictor-rank candidates actually evaluated for one
    component count, sorted by ascending predictor rank. The selected scalar
    result follows the fitted search score and tie-breaking rule. Under the
    default scorer, that selection minimizes mean response-standardized CV-MSE.

    Attributes
    ----------
    n_components : int
        Component count shared by every evaluated candidate.
    predictor_rank : ndarray of shape (n_evaluated_ranks,)
        Evaluated predictor ranks in strictly ascending order.
    mean_test_score : ndarray of shape (n_evaluated_ranks,)
        Mean configured test score for each evaluated rank.
    cv_mse_mean : ndarray of shape (n_evaluated_ranks,)
        Mean response-standardized validation MSE for each evaluated rank.
    cv_mse_fold_sd : ndarray of shape (n_evaluated_ranks,)
        Population standard deviation of response-standardized MSE across folds.
    n_splits : int
        Number of cross-validation splits.
    selected : PiPLSComponentResult
        Conditionally selected scalar result for ``n_components``.
    """

    n_components: int
    predictor_rank: IntArray
    mean_test_score: FloatArray
    cv_mse_mean: FloatArray
    cv_mse_fold_sd: FloatArray
    n_splits: int
    selected: PiPLSComponentResult

    def __post_init__(self) -> None:
        n_components = _positive_python_int(self.n_components, name="n_components")
        predictor_rank = _read_only_int_array(
            self.predictor_rank,
            name="predictor_rank",
        )
        mean_test_score = _read_only_float_array(
            self.mean_test_score,
            name="mean_test_score",
        )
        cv_mse_mean = _read_only_float_array(self.cv_mse_mean, name="cv_mse_mean")
        cv_mse_fold_sd = _read_only_float_array(
            self.cv_mse_fold_sd,
            name="cv_mse_fold_sd",
        )
        n_splits = _positive_python_int(self.n_splits, name="n_splits")

        arrays = (mean_test_score, cv_mse_mean, cv_mse_fold_sd)
        if predictor_rank.size == 0:
            raise ValueError("A predictor-rank profile must contain at least one result.")
        if any(array.size != predictor_rank.size for array in arrays):
            raise ValueError("All predictor-rank-profile arrays must have the same length.")
        if np.any(predictor_rank < n_components):
            raise ValueError(
                "predictor_rank must not be smaller than n_components."
            )
        if np.any(np.diff(predictor_rank) <= 0):
            raise ValueError("predictor_rank must be unique and strictly ascending.")
        if any(not np.all(np.isfinite(array)) for array in arrays):
            raise ValueError("Predictor-rank-profile score arrays must contain finite values.")
        if np.any(cv_mse_fold_sd < 0.0):
            raise ValueError("cv_mse_fold_sd must contain nonnegative values.")
        if not isinstance(self.selected, PiPLSComponentResult):
            raise ValueError("selected must be a PiPLSComponentResult.")
        if self.selected.n_components != n_components:
            raise ValueError("selected.n_components must match n_components.")
        if self.selected.n_splits != n_splits:
            raise ValueError("selected.n_splits must match n_splits.")

        selected_rows = np.flatnonzero(predictor_rank == self.selected.predictor_rank)
        if selected_rows.size != 1:
            raise ValueError("selected.predictor_rank must occur exactly once in predictor_rank.")
        selected_index = int(selected_rows[0])
        selected_values = (
            (self.selected.mean_test_score, mean_test_score[selected_index]),
            (self.selected.cv_mse_mean, cv_mse_mean[selected_index]),
            (self.selected.cv_mse_fold_sd, cv_mse_fold_sd[selected_index]),
        )
        if any(
            not np.isclose(scalar, array_value, rtol=1e-7, atol=1e-12)
            for scalar, array_value in selected_values
        ):
            raise ValueError("selected score values must match its predictor-rank row.")

        object.__setattr__(self, "n_components", n_components)
        object.__setattr__(self, "predictor_rank", predictor_rank)
        object.__setattr__(self, "mean_test_score", mean_test_score)
        object.__setattr__(self, "cv_mse_mean", cv_mse_mean)
        object.__setattr__(self, "cv_mse_fold_sd", cv_mse_fold_sd)
        object.__setattr__(self, "n_splits", n_splits)

    def __reduce__(self) -> tuple[type[PiPLSPredictorRankProfile], tuple[object, ...]]:
        """Reconstruct through validation so unpickled arrays remain read-only."""

        return (
            type(self),
            (
                self.n_components,
                self.predictor_rank,
                self.mean_test_score,
                self.cv_mse_mean,
                self.cv_mse_fold_sd,
                self.n_splits,
                self.selected,
            ),
        )


@dataclass(frozen=True)
class PiPLSComponentPath:
    """Immutable concise component-path results.

    Each array contains one conditionally selected predictor-rank result per
    evaluated component count. Arrays are defensive, read-only copies and are
    aligned by row.

    Attributes
    ----------
    n_components : ndarray of shape (n_component_values,)
        Evaluated component counts in strictly ascending order.
    predictor_rank : ndarray of shape (n_component_values,)
        Conditionally selected predictor rank for each component count.
    predictor_rank_policy : ndarray of shape (n_component_values,)
        Predictor-rank policy for each row.
    mean_test_score : ndarray of shape (n_component_values,)
        Mean configured test score for each selected candidate.
    cv_mse_mean : ndarray of shape (n_component_values,)
        Mean response-standardized validation MSE for each selected candidate.
    cv_mse_fold_sd : ndarray of shape (n_component_values,)
        Population standard deviation of response-standardized MSE across folds.
    n_splits : ndarray of shape (n_component_values,)
        Number of cross-validation splits represented by each row.
    """

    n_components: IntArray
    predictor_rank: IntArray
    predictor_rank_policy: StringArray
    mean_test_score: FloatArray
    cv_mse_mean: FloatArray
    cv_mse_fold_sd: FloatArray
    n_splits: IntArray

    def __post_init__(self) -> None:
        n_components = _read_only_int_array(self.n_components, name="n_components")
        predictor_rank = _read_only_int_array(
            self.predictor_rank,
            name="predictor_rank",
        )
        predictor_rank_policy = _read_only_policy_array(self.predictor_rank_policy)
        mean_test_score = _read_only_float_array(
            self.mean_test_score,
            name="mean_test_score",
        )
        cv_mse_mean = _read_only_float_array(self.cv_mse_mean, name="cv_mse_mean")
        cv_mse_fold_sd = _read_only_float_array(
            self.cv_mse_fold_sd,
            name="cv_mse_fold_sd",
        )
        n_splits = _read_only_int_array(self.n_splits, name="n_splits")

        arrays = (
            predictor_rank,
            predictor_rank_policy,
            mean_test_score,
            cv_mse_mean,
            cv_mse_fold_sd,
            n_splits,
        )
        if n_components.size == 0:
            raise ValueError("A component path must contain at least one result.")
        if any(array.size != n_components.size for array in arrays):
            raise ValueError("All component-path arrays must have the same length.")
        if np.any(n_components <= 0):
            raise ValueError("n_components must contain positive integers.")
        if np.any(np.diff(n_components) <= 0):
            raise ValueError("n_components must be unique and strictly ascending.")
        if np.any(predictor_rank < n_components):
            raise ValueError(
                "predictor_rank must not be smaller than the aligned n_components value."
            )
        if np.any(n_splits <= 0):
            raise ValueError("n_splits must contain positive integers.")
        if np.any(cv_mse_fold_sd < 0.0):
            raise ValueError("cv_mse_fold_sd must contain nonnegative values.")

        object.__setattr__(self, "n_components", n_components)
        object.__setattr__(self, "predictor_rank", predictor_rank)
        object.__setattr__(self, "predictor_rank_policy", predictor_rank_policy)
        object.__setattr__(self, "mean_test_score", mean_test_score)
        object.__setattr__(self, "cv_mse_mean", cv_mse_mean)
        object.__setattr__(self, "cv_mse_fold_sd", cv_mse_fold_sd)
        object.__setattr__(self, "n_splits", n_splits)

    def __reduce__(self) -> tuple[type[PiPLSComponentPath], tuple[object, ...]]:
        """Reconstruct through validation so unpickled arrays remain read-only."""

        return (
            type(self),
            (
                self.n_components,
                self.predictor_rank,
                self.predictor_rank_policy,
                self.mean_test_score,
                self.cv_mse_mean,
                self.cv_mse_fold_sd,
                self.n_splits,
            ),
        )

    def for_n_components(self, n_components: int) -> PiPLSComponentResult:
        """Return the selected result for one evaluated component count.

        Parameters
        ----------
        n_components : int
            Component count to retrieve.

        Returns
        -------
        PiPLSComponentResult
            Frozen scalar result for the requested component count.

        Raises
        ------
        ValueError
            If ``n_components`` is not an integer or was not evaluated.
        """

        if isinstance(n_components, bool) or not isinstance(
            n_components,
            (int, np.integer),
        ):
            raise ValueError(
                "n_components must be an integer present in the fitted component path."
            )
        requested = int(n_components)
        index = int(np.searchsorted(self.n_components, requested))
        if index >= self.n_components.size or int(self.n_components[index]) != requested:
            available = ", ".join(str(int(value)) for value in self.n_components)
            raise ValueError(
                f"n_components={requested} was not evaluated. Available values are "
                f"[{available}]."
            )
        return PiPLSComponentResult(
            n_components=int(self.n_components[index]),
            predictor_rank=int(self.predictor_rank[index]),
            predictor_rank_policy=cast(
                PredictorRankPolicy,
                str(self.predictor_rank_policy[index]),
            ),
            mean_test_score=float(self.mean_test_score[index]),
            cv_mse_mean=float(self.cv_mse_mean[index]),
            cv_mse_fold_sd=float(self.cv_mse_fold_sd[index]),
            n_splits=int(self.n_splits[index]),
        )


def _read_only_int_array(value: ArrayLike, *, name: str) -> IntArray:
    array = np.array(value, dtype=np.intp, copy=True)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional; got shape {array.shape}.")
    array.setflags(write=False)
    return array


def _read_only_float_array(value: ArrayLike, *, name: str) -> FloatArray:
    array = np.array(value, dtype=np.float64, copy=True)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional; got shape {array.shape}.")
    array.setflags(write=False)
    return array


def _read_only_policy_array(value: ArrayLike) -> StringArray:
    array = np.array(value, dtype=np.str_, copy=True)
    if array.ndim != 1:
        raise ValueError(
            "predictor_rank_policy must be one-dimensional; "
            f"got shape {array.shape}."
        )
    invalid = sorted(set(array.tolist()) - _ALLOWED_PREDICTOR_RANK_POLICIES)
    if invalid:
        raise ValueError(
            "predictor_rank_policy contains unsupported values: "
            f"{invalid}."
        )
    array.setflags(write=False)
    return array


def _positive_python_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be a positive integer.")
    converted = int(value)
    if converted <= 0:
        raise ValueError(f"{name} must be a positive integer.")
    return converted
