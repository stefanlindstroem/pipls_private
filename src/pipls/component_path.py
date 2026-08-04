"""Immutable concise results for a fitted Pi-PLS component path."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Real
from typing import Literal, cast

import numpy as np
from numpy.typing import NDArray

from ._model_selection import _tied_score_mask
from ._result_validation import (
    _finite_float,
    _literal_string,
    _nonnegative_finite_float,
    _positive_int,
    _read_only_float_array,
    _read_only_int_array,
)

__all__ = [
    "PiPLSComponentPath",
    "PiPLSPredictorRankProfile",
    "PiPLSSelection",
]

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.intp]
SelectionRule = Literal["best_score", "minimum_cv_mse"]
PredictorRankPolicy = Literal["optimized", "fixed", "maximum"]
_ALLOWED_SELECTION_RULES = frozenset({"best_score", "minimum_cv_mse"})
_ALLOWED_PREDICTOR_RANK_POLICIES = frozenset({"optimized", "fixed", "maximum"})


def _optional_relative_tolerance(value: object) -> float | None:
    """Validate optional finite nonnegative relative tolerance provenance."""

    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError("relative_tolerance must be a finite nonnegative real number.")
    converted = float(value)
    if not np.isfinite(converted) or converted < 0.0:
        raise ValueError("relative_tolerance must be a finite nonnegative real number.")
    return converted


def _optional_absolute_tolerance(value: object) -> float | None:
    """Validate optional nonnegative absolute tolerance provenance."""

    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(
            "absolute_tolerance must be a nonnegative real number or positive infinity."
        )
    converted = float(value)
    if np.isnan(converted) or converted < 0.0:
        raise ValueError(
            "absolute_tolerance must be a nonnegative real number or positive infinity."
        )
    return converted


def _cv_mse_tolerance_threshold(
    minimum_cv_mse: float,
    relative_tolerance: float,
    absolute_tolerance: float,
) -> float:
    """Return the simultaneous relative-and-absolute CV-MSE threshold."""

    with np.errstate(over="ignore", invalid="ignore"):
        relative_threshold = np.float64(minimum_cv_mse) * np.float64(
            1.0 + relative_tolerance
        )
        absolute_threshold = np.float64(minimum_cv_mse) + np.float64(
            absolute_tolerance
        )
    return float(np.minimum(relative_threshold, absolute_threshold))


@dataclass(frozen=True)
class PiPLSSelection:
    """Immutable Pi-PLS selection for one evaluated rank pair.

    Attributes
    ----------
    n_components : int
        Evaluated number of paired latent modes.
    predictor_rank : int
        Predictor rank selected conditionally for ``n_components``.
    predictor_rank_policy : {"optimized", "fixed", "maximum"}
        Interpretation of the predictor-rank specification used by the path.
    mean_test_score : float
        Mean configured test score for the selected candidate.
    cv_mse_mean : float
        Mean response-standardized validation MSE.
    cv_mse_std : float
        Population standard deviation of response-standardized MSE across
        validation splits.
    n_splits : int
        Number of cross-validation splits.
    rule : {"best_score", "minimum_cv_mse"} or None
        Search-owned rule that produced this selection. ``None`` denotes direct
        lookup by component count.
    reference_minimum : PiPLSSelection or None
        Exact minimum-CV-MSE path row used to derive a tolerance selection.
        The reference row has no rule provenance.
    relative_tolerance : float or None
        Resolved nonnegative relative tolerance for ``rule="minimum_cv_mse"``.
    absolute_tolerance : float or None
        Resolved nonnegative absolute tolerance for ``rule="minimum_cv_mse"``;
        positive infinity disables the absolute cap.
    cv_mse_threshold : float or None
        Derived effective threshold for ``rule="minimum_cv_mse"``.
    """

    n_components: int
    predictor_rank: int
    predictor_rank_policy: PredictorRankPolicy
    mean_test_score: float
    cv_mse_mean: float
    cv_mse_std: float
    n_splits: int
    rule: SelectionRule | None = None
    reference_minimum: PiPLSSelection | None = None
    relative_tolerance: float | None = None
    absolute_tolerance: float | None = None

    def __post_init__(self) -> None:
        n_components = _positive_int(self.n_components, name="n_components")
        predictor_rank = _positive_int(self.predictor_rank, name="predictor_rank")
        if n_components > predictor_rank:
            raise ValueError("n_components must not exceed predictor_rank.")
        predictor_rank_policy = cast(
            PredictorRankPolicy,
            _literal_string(
                self.predictor_rank_policy,
                name="predictor_rank_policy",
                allowed=_ALLOWED_PREDICTOR_RANK_POLICIES,
            ),
        )
        mean_test_score = _finite_float(self.mean_test_score, name="mean_test_score")
        cv_mse_mean = _nonnegative_finite_float(self.cv_mse_mean, name="cv_mse_mean")
        cv_mse_std = _nonnegative_finite_float(
            self.cv_mse_std,
            name="cv_mse_std",
        )
        n_splits = _positive_int(self.n_splits, name="n_splits")
        rule = (
            None
            if self.rule is None
            else cast(
                SelectionRule,
                _literal_string(
                    self.rule,
                    name="rule",
                    allowed=_ALLOWED_SELECTION_RULES,
                ),
            )
        )
        reference_minimum = self.reference_minimum
        if reference_minimum is not None and not isinstance(
            reference_minimum,
            PiPLSSelection,
        ):
            raise TypeError("reference_minimum must be a PiPLSSelection or None.")
        relative_tolerance = _optional_relative_tolerance(self.relative_tolerance)
        absolute_tolerance = _optional_absolute_tolerance(self.absolute_tolerance)

        if rule == "minimum_cv_mse":
            if reference_minimum is None:
                raise ValueError(
                    f"reference_minimum is required for {rule!r} selection."
                )
            if reference_minimum.rule is not None:
                raise ValueError("reference_minimum must be an unruled path row.")
            if reference_minimum.reference_minimum is not None:
                raise ValueError(
                    "reference_minimum must not contain nested provenance."
                )
            if (
                reference_minimum.relative_tolerance is not None
                or reference_minimum.absolute_tolerance is not None
            ):
                raise ValueError(
                    "reference_minimum must not contain tolerance provenance."
                )
            if reference_minimum.predictor_rank_policy != predictor_rank_policy:
                raise ValueError(
                    "reference_minimum must use the same predictor-rank policy."
                )
            if reference_minimum.n_splits != n_splits:
                raise ValueError(
                    "reference_minimum must use the same validation split count."
                )
            if reference_minimum.cv_mse_mean > cv_mse_mean:
                raise ValueError(
                    "reference_minimum CV-MSE must not exceed the selected CV-MSE."
                )

        if rule == "minimum_cv_mse":
            if relative_tolerance is None or absolute_tolerance is None:
                raise ValueError(
                    "relative_tolerance and absolute_tolerance are required for "
                    'rule="minimum_cv_mse".'
                )
            reference = cast(PiPLSSelection, reference_minimum)
            threshold = _cv_mse_tolerance_threshold(
                reference.cv_mse_mean,
                relative_tolerance,
                absolute_tolerance,
            )
            if cv_mse_mean > threshold:
                raise ValueError(
                    "Selected CV-MSE must not exceed the effective CV-MSE threshold."
                )
        else:
            if reference_minimum is not None:
                raise ValueError(
                    "reference_minimum is defined only for minimum-CV-MSE selection."
                )
            if relative_tolerance is not None or absolute_tolerance is not None:
                raise ValueError(
                    "Tolerance provenance is defined only for minimum-CV-MSE selection."
                )

        object.__setattr__(self, "n_components", n_components)
        object.__setattr__(self, "predictor_rank", predictor_rank)
        object.__setattr__(self, "predictor_rank_policy", predictor_rank_policy)
        object.__setattr__(self, "mean_test_score", mean_test_score)
        object.__setattr__(self, "cv_mse_mean", cv_mse_mean)
        object.__setattr__(self, "cv_mse_std", cv_mse_std)
        object.__setattr__(self, "n_splits", n_splits)
        object.__setattr__(self, "rule", rule)
        object.__setattr__(self, "reference_minimum", reference_minimum)
        object.__setattr__(self, "relative_tolerance", relative_tolerance)
        object.__setattr__(self, "absolute_tolerance", absolute_tolerance)

    @property
    def cv_mse_threshold(self) -> float | None:
        """Return the effective threshold for minimum-CV-MSE selection."""

        if self.rule != "minimum_cv_mse":
            return None
        reference = self.reference_minimum
        relative_tolerance = self.relative_tolerance
        absolute_tolerance = self.absolute_tolerance
        if (
            reference is None
            or relative_tolerance is None
            or absolute_tolerance is None
        ):  # pragma: no cover - guarded by construction
            raise RuntimeError(
                "A minimum-CV-MSE selection requires complete tolerance provenance."
            )
        return _cv_mse_tolerance_threshold(
            reference.cv_mse_mean,
            relative_tolerance,
            absolute_tolerance,
        )

    def __reduce__(self) -> tuple[type[PiPLSSelection], tuple[object, ...]]:
        """Reconstruct through validation during unpickling."""

        return (
            type(self),
            (
                self.n_components,
                self.predictor_rank,
                self.predictor_rank_policy,
                self.mean_test_score,
                self.cv_mse_mean,
                self.cv_mse_std,
                self.n_splits,
                self.rule,
                self.reference_minimum,
                self.relative_tolerance,
                self.absolute_tolerance,
            ),
        )


@dataclass(frozen=True)
class PiPLSPredictorRankProfile:
    """Immutable evaluated predictor-rank results for one paired-mode count.

    Arrays contain the predictor-rank candidates actually evaluated for one
    paired-mode count, sorted by ascending predictor rank. The selected scalar
    result follows the fitted search score and tie-breaking rule. Under the
    default scorer, that selection minimizes mean response-standardized CV-MSE.

    Attributes
    ----------
    n_components : int
        Number of paired latent modes shared by every evaluated candidate.
    predictor_rank : ndarray of shape (n_evaluated_ranks,)
        Evaluated predictor ranks in strictly ascending order.
    mean_test_score : ndarray of shape (n_evaluated_ranks,)
        Mean configured test score for each evaluated rank.
    cv_mse_mean : ndarray of shape (n_evaluated_ranks,)
        Mean response-standardized validation MSE for each evaluated rank.
    cv_mse_std : ndarray of shape (n_evaluated_ranks,)
        Population standard deviation of response-standardized MSE across
        validation splits.
    predictor_rank_policy : {"optimized", "fixed", "maximum"}
        Predictor-rank policy shared by every evaluated candidate.
    n_splits : int
        Number of cross-validation splits.
    selection : PiPLSSelection
        Derived conditional selection for ``n_components``.
    """

    n_components: int
    predictor_rank: IntArray
    mean_test_score: FloatArray
    cv_mse_mean: FloatArray
    cv_mse_std: FloatArray
    predictor_rank_policy: PredictorRankPolicy
    n_splits: int

    def __post_init__(self) -> None:
        n_components = _positive_int(self.n_components, name="n_components")
        predictor_rank = _read_only_int_array(
            self.predictor_rank,
            name="predictor_rank",
        )
        mean_test_score = _read_only_float_array(
            self.mean_test_score,
            name="mean_test_score",
        )
        cv_mse_mean = _read_only_float_array(self.cv_mse_mean, name="cv_mse_mean")
        cv_mse_std = _read_only_float_array(
            self.cv_mse_std,
            name="cv_mse_std",
        )
        predictor_rank_policy = cast(
            PredictorRankPolicy,
            _literal_string(
                self.predictor_rank_policy,
                name="predictor_rank_policy",
                allowed=_ALLOWED_PREDICTOR_RANK_POLICIES,
            ),
        )
        n_splits = _positive_int(self.n_splits, name="n_splits")

        arrays = (mean_test_score, cv_mse_mean, cv_mse_std)
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
        if np.any(cv_mse_mean < 0.0):
            raise ValueError("cv_mse_mean must contain nonnegative values.")
        if np.any(cv_mse_std < 0.0):
            raise ValueError("cv_mse_std must contain nonnegative values.")

        object.__setattr__(self, "n_components", n_components)
        object.__setattr__(self, "predictor_rank", predictor_rank)
        object.__setattr__(self, "mean_test_score", mean_test_score)
        object.__setattr__(self, "cv_mse_mean", cv_mse_mean)
        object.__setattr__(self, "cv_mse_std", cv_mse_std)
        object.__setattr__(self, "predictor_rank_policy", predictor_rank_policy)
        object.__setattr__(self, "n_splits", n_splits)

    @property
    def selection(self) -> PiPLSSelection:
        """Return the conditional predictor-rank selection."""

        maximum = float(np.max(self.mean_test_score))
        tied = np.flatnonzero(_tied_score_mask(self.mean_test_score, maximum))
        index = int(tied[0])
        return PiPLSSelection(
            n_components=self.n_components,
            predictor_rank=int(self.predictor_rank[index]),
            predictor_rank_policy=self.predictor_rank_policy,
            mean_test_score=float(self.mean_test_score[index]),
            cv_mse_mean=float(self.cv_mse_mean[index]),
            cv_mse_std=float(self.cv_mse_std[index]),
            n_splits=self.n_splits,
        )

    def __reduce__(self) -> tuple[type[PiPLSPredictorRankProfile], tuple[object, ...]]:
        """Reconstruct through validation so unpickled arrays remain read-only."""

        return (
            type(self),
            (
                self.n_components,
                self.predictor_rank,
                self.mean_test_score,
                self.cv_mse_mean,
                self.cv_mse_std,
                self.predictor_rank_policy,
                self.n_splits,
            ),
        )


@dataclass(frozen=True)
class PiPLSComponentPath:
    """Immutable concise component-path results.

    Each array contains one conditionally selected predictor-rank result per
    evaluated paired-mode count. Arrays are defensive, read-only copies and are
    aligned by row.

    Attributes
    ----------
    n_components : ndarray of shape (n_component_values,)
        Evaluated paired-mode counts in strictly ascending order.
    predictor_rank : ndarray of shape (n_component_values,)
        Conditionally selected predictor rank for each paired-mode count.
    predictor_rank_policy : {"optimized", "fixed", "maximum"}
        Predictor-rank policy shared by every path row.
    mean_test_score : ndarray of shape (n_component_values,)
        Mean configured test score for each selected candidate.
    cv_mse_mean : ndarray of shape (n_component_values,)
        Mean response-standardized validation MSE for each selected candidate.
    cv_mse_std : ndarray of shape (n_component_values,)
        Population standard deviation of response-standardized MSE across
        validation splits.
    n_splits : int
        Number of cross-validation splits shared by every path row.
    """

    n_components: IntArray
    predictor_rank: IntArray
    predictor_rank_policy: PredictorRankPolicy
    mean_test_score: FloatArray
    cv_mse_mean: FloatArray
    cv_mse_std: FloatArray
    n_splits: int

    def __post_init__(self) -> None:
        n_components = _read_only_int_array(self.n_components, name="n_components")
        predictor_rank = _read_only_int_array(
            self.predictor_rank,
            name="predictor_rank",
        )
        predictor_rank_policy = cast(
            PredictorRankPolicy,
            _literal_string(
                self.predictor_rank_policy,
                name="predictor_rank_policy",
                allowed=_ALLOWED_PREDICTOR_RANK_POLICIES,
            ),
        )
        mean_test_score = _read_only_float_array(
            self.mean_test_score,
            name="mean_test_score",
        )
        cv_mse_mean = _read_only_float_array(self.cv_mse_mean, name="cv_mse_mean")
        cv_mse_std = _read_only_float_array(
            self.cv_mse_std,
            name="cv_mse_std",
        )
        n_splits = _positive_int(self.n_splits, name="n_splits")

        arrays = (
            predictor_rank,
            mean_test_score,
            cv_mse_mean,
            cv_mse_std,
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
        if np.any(cv_mse_mean < 0.0):
            raise ValueError("cv_mse_mean must contain nonnegative values.")
        if np.any(cv_mse_std < 0.0):
            raise ValueError("cv_mse_std must contain nonnegative values.")

        object.__setattr__(self, "n_components", n_components)
        object.__setattr__(self, "predictor_rank", predictor_rank)
        object.__setattr__(self, "predictor_rank_policy", predictor_rank_policy)
        object.__setattr__(self, "mean_test_score", mean_test_score)
        object.__setattr__(self, "cv_mse_mean", cv_mse_mean)
        object.__setattr__(self, "cv_mse_std", cv_mse_std)
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
                self.cv_mse_std,
                self.n_splits,
            ),
        )

    def _selection_at_index(self, index: int) -> PiPLSSelection:
        """Return one selection from an internally validated row index."""

        return PiPLSSelection(
            n_components=int(self.n_components[index]),
            predictor_rank=int(self.predictor_rank[index]),
            predictor_rank_policy=self.predictor_rank_policy,
            mean_test_score=float(self.mean_test_score[index]),
            cv_mse_mean=float(self.cv_mse_mean[index]),
            cv_mse_std=float(self.cv_mse_std[index]),
            n_splits=self.n_splits,
        )
