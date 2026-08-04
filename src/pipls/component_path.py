"""Immutable concise results for a fitted Pi-PLS component path."""

from __future__ import annotations

from dataclasses import dataclass
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
SelectionRule = Literal["best_score", "minimum_cv_mse", "one_standard_error"]
PredictorRankPolicy = Literal["optimized", "fixed", "maximum"]
_ALLOWED_SELECTION_RULES = frozenset(
    {"best_score", "minimum_cv_mse", "one_standard_error"}
)
_ALLOWED_PREDICTOR_RANK_POLICIES = frozenset({"optimized", "fixed", "maximum"})


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
    cv_mse_fold_sd : float
        Population standard deviation of response-standardized MSE across folds.
    cv_mse_standard_error : float
        Fold-based standard error of the mean response-standardized CV-MSE.
    n_splits : int
        Number of cross-validation splits.
    rule : {"best_score", "minimum_cv_mse", "one_standard_error"} or None
        Search-owned rule that produced this selection. ``None`` denotes direct
        lookup by component count.
    reference_minimum : PiPLSSelection or None
        Minimum-CV-MSE selection used to derive a one-standard-error selection.
        Defined only when ``rule="one_standard_error"``.
    one_standard_error_threshold : float or None
        Derived minimum-CV-MSE plus its fold-based standard error. Defined only
        when ``rule="one_standard_error"``.
    """

    n_components: int
    predictor_rank: int
    predictor_rank_policy: PredictorRankPolicy
    mean_test_score: float
    cv_mse_mean: float
    cv_mse_fold_sd: float
    n_splits: int
    rule: SelectionRule | None = None
    reference_minimum: PiPLSSelection | None = None

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
        cv_mse_fold_sd = _nonnegative_finite_float(
            self.cv_mse_fold_sd,
            name="cv_mse_fold_sd",
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
            raise TypeError(
                "reference_minimum must be a PiPLSSelection or None."
            )

        if rule == "one_standard_error":
            if reference_minimum is None:
                raise ValueError(
                    "reference_minimum is required for one-standard-error selection."
                )
            if reference_minimum.rule != "minimum_cv_mse":
                raise ValueError(
                    'reference_minimum must use rule="minimum_cv_mse".'
                )
            if reference_minimum.reference_minimum is not None:
                raise ValueError("reference_minimum must not contain nested provenance.")
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
            threshold = (
                reference_minimum.cv_mse_mean
                + reference_minimum.cv_mse_standard_error
            )
            if not np.isfinite(threshold):
                raise ValueError("The one-standard-error threshold must be finite.")
            if cv_mse_mean > threshold:
                raise ValueError(
                    "Selected CV-MSE must not exceed the one-standard-error threshold."
                )
        elif reference_minimum is not None:
            raise ValueError(
                "reference_minimum is defined only for one-standard-error selection."
            )

        object.__setattr__(self, "n_components", n_components)
        object.__setattr__(self, "predictor_rank", predictor_rank)
        object.__setattr__(self, "predictor_rank_policy", predictor_rank_policy)
        object.__setattr__(self, "mean_test_score", mean_test_score)
        object.__setattr__(self, "cv_mse_mean", cv_mse_mean)
        object.__setattr__(self, "cv_mse_fold_sd", cv_mse_fold_sd)
        object.__setattr__(self, "n_splits", n_splits)
        object.__setattr__(self, "rule", rule)
        object.__setattr__(self, "reference_minimum", reference_minimum)

    @property
    def cv_mse_standard_error(self) -> float:
        """Return the fold-based standard error of mean CV-MSE.

        ``cv_mse_fold_sd`` stores a population standard deviation. Dividing it
        by ``sqrt(n_splits - 1)`` is equivalent to converting it to the sample
        standard deviation and then dividing by ``sqrt(n_splits)``.
        """

        if self.n_splits < 2:
            raise ValueError(
                "cv_mse_standard_error requires at least two validation splits."
            )
        return float(self.cv_mse_fold_sd / np.sqrt(self.n_splits - 1))

    @property
    def one_standard_error_threshold(self) -> float | None:
        """Return the exact threshold used by one-standard-error selection."""

        if self.rule != "one_standard_error":
            return None
        reference = self.reference_minimum
        if reference is None:  # pragma: no cover - guarded by construction
            raise RuntimeError(
                "A one-standard-error selection requires a reference minimum."
            )
        return float(reference.cv_mse_mean + reference.cv_mse_standard_error)

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
                self.cv_mse_fold_sd,
                self.n_splits,
                self.rule,
                self.reference_minimum,
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
    cv_mse_fold_sd : ndarray of shape (n_evaluated_ranks,)
        Population standard deviation of response-standardized MSE across folds.
    cv_mse_standard_error : ndarray of shape (n_evaluated_ranks,)
        Fold-based standard error of mean response-standardized CV-MSE.
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
    cv_mse_fold_sd: FloatArray
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
        cv_mse_fold_sd = _read_only_float_array(
            self.cv_mse_fold_sd,
            name="cv_mse_fold_sd",
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
        if np.any(cv_mse_mean < 0.0):
            raise ValueError("cv_mse_mean must contain nonnegative values.")
        if np.any(cv_mse_fold_sd < 0.0):
            raise ValueError("cv_mse_fold_sd must contain nonnegative values.")

        object.__setattr__(self, "n_components", n_components)
        object.__setattr__(self, "predictor_rank", predictor_rank)
        object.__setattr__(self, "mean_test_score", mean_test_score)
        object.__setattr__(self, "cv_mse_mean", cv_mse_mean)
        object.__setattr__(self, "cv_mse_fold_sd", cv_mse_fold_sd)
        object.__setattr__(self, "predictor_rank_policy", predictor_rank_policy)
        object.__setattr__(self, "n_splits", n_splits)

    @property
    def cv_mse_standard_error(self) -> FloatArray:
        """Return read-only fold-based standard errors of mean CV-MSE."""

        return _read_only_cv_mse_standard_error(
            self.cv_mse_fold_sd,
            self.n_splits,
        )

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
            cv_mse_fold_sd=float(self.cv_mse_fold_sd[index]),
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
                self.cv_mse_fold_sd,
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
    cv_mse_fold_sd : ndarray of shape (n_component_values,)
        Population standard deviation of response-standardized MSE across folds.
    cv_mse_standard_error : ndarray of shape (n_component_values,)
        Fold-based standard error of mean response-standardized CV-MSE.
    n_splits : int
        Number of cross-validation splits shared by every path row.
    """

    n_components: IntArray
    predictor_rank: IntArray
    predictor_rank_policy: PredictorRankPolicy
    mean_test_score: FloatArray
    cv_mse_mean: FloatArray
    cv_mse_fold_sd: FloatArray
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
        cv_mse_fold_sd = _read_only_float_array(
            self.cv_mse_fold_sd,
            name="cv_mse_fold_sd",
        )
        n_splits = _positive_int(self.n_splits, name="n_splits")

        arrays = (
            predictor_rank,
            mean_test_score,
            cv_mse_mean,
            cv_mse_fold_sd,
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
        if np.any(cv_mse_fold_sd < 0.0):
            raise ValueError("cv_mse_fold_sd must contain nonnegative values.")

        object.__setattr__(self, "n_components", n_components)
        object.__setattr__(self, "predictor_rank", predictor_rank)
        object.__setattr__(self, "predictor_rank_policy", predictor_rank_policy)
        object.__setattr__(self, "mean_test_score", mean_test_score)
        object.__setattr__(self, "cv_mse_mean", cv_mse_mean)
        object.__setattr__(self, "cv_mse_fold_sd", cv_mse_fold_sd)
        object.__setattr__(self, "n_splits", n_splits)

    @property
    def cv_mse_standard_error(self) -> FloatArray:
        """Return read-only fold-based standard errors of mean CV-MSE."""

        return _read_only_cv_mse_standard_error(
            self.cv_mse_fold_sd,
            self.n_splits,
        )

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

    def _selection_at_index(self, index: int) -> PiPLSSelection:
        """Return one selection from an internally validated row index."""

        return PiPLSSelection(
            n_components=int(self.n_components[index]),
            predictor_rank=int(self.predictor_rank[index]),
            predictor_rank_policy=self.predictor_rank_policy,
            mean_test_score=float(self.mean_test_score[index]),
            cv_mse_mean=float(self.cv_mse_mean[index]),
            cv_mse_fold_sd=float(self.cv_mse_fold_sd[index]),
            n_splits=self.n_splits,
        )


def _read_only_cv_mse_standard_error(
    cv_mse_fold_sd: FloatArray,
    n_splits: IntArray | int,
) -> FloatArray:
    """Derive sample-standard-error values from stored population fold SDs."""

    split_counts = np.asarray(n_splits, dtype=np.float64)
    if np.any(split_counts < 2.0):
        raise ValueError(
            "cv_mse_standard_error requires at least two validation splits."
        )
    standard_error = np.asarray(
        cv_mse_fold_sd / np.sqrt(split_counts - 1.0),
        dtype=np.float64,
    )
    standard_error.setflags(write=False)
    return standard_error
