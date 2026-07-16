"""Public Pi-PLS estimator with fixed, exhaustive, and adaptive rank modes."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal, cast

import numpy as np
from joblib import Parallel, delayed
from numpy.typing import ArrayLike, NDArray
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin
from sklearn.metrics import get_scorer, r2_score
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

from ._core import fit_pipls_core
from .metrics import neg_response_standardized_mean_squared_error
from .model_selection import (
    _ADAPTIVE_EXHAUSTIVE_THRESHOLD,
    _adaptive_refinement_interval,
    _as_positive_float,
    _logarithmic_predictor_rank_values,
    _materialize_cv_splits,
    _max_predictor_rank,
    _predictor_rank_values,
    _response_standardized_mse,
    _select_predictor_rank,
    _training_response_scale,
)

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.intp]
Scorer = Callable[[Any, ArrayLike, ArrayLike], float]
_DEFAULT_SCORING = "neg_response_standardized_mean_squared_error"


@dataclass(frozen=True)
class _CandidateCVResult:
    """Cross-validation results for one predictor-rank candidate."""

    predictor_rank: int
    split_scores: FloatArray
    split_response_standardized_mse: FloatArray


class PiPLSRegression(TransformerMixin, RegressorMixin, BaseEstimator):  # type: ignore[misc]
    r"""Pi-PLS regression with fixed, rule-derived, exhaustive, or adaptive rank.

    Parameters
    ----------
    n_components:
        Response-side latent dimension $h$.
    scale:
        If true, center and divide predictor and response columns by their
        training-sample standard deviations. If false, center without scaling.
    copy:
        If true, copy input arrays before preprocessing.
    predictor_rank:
        Predictor truncation rank $r_\pi$. An integer fixes the rank explicitly,
        ``"max"`` uses the rule-derived upper bound directly, ``"optimal"``
        exhaustively searches all admissible ranks, and ``"auto"`` uses a
        deterministic adaptive coarse-to-fine search.
    samples_per_predictor_rank:
        Positive rule parameter $c$ used to derive the upper predictor rank.
        It does not constrain an explicitly supplied integer rank.
    cv:
        Cross-validation splitter, split count, or iterable of train-validation
        index pairs used when ``predictor_rank`` is ``"auto"`` or ``"optimal"``.
    scoring:
        Scikit-learn scorer name or callable used only in cross-validated modes.
        The default is negative response-standardized mean squared error.
    n_jobs:
        Number of predictor-rank candidates evaluated concurrently in
        cross-validated modes. ``None`` uses joblib's default and ``-1`` uses all
        processors.
    """

    def __init__(
        self,
        n_components: int = 2,
        *,
        scale: bool = True,
        copy: bool = True,
        predictor_rank: int | Literal["max", "optimal", "auto"] = "auto",
        samples_per_predictor_rank: float = 10.0,
        cv: object = 5,
        scoring: str | Scorer = _DEFAULT_SCORING,
        n_jobs: int | None = None,
    ) -> None:
        self.n_components = n_components
        self.scale = scale
        self.copy = copy
        self.predictor_rank = predictor_rank
        self.samples_per_predictor_rank = samples_per_predictor_rank
        self.cv = cv
        self.scoring = scoring
        self.n_jobs = n_jobs

    def fit(self, X: ArrayLike, y: ArrayLike) -> PiPLSRegression:
        """Fit the Pi-PLS model and select predictor rank when requested."""

        X_checked, y_checked = check_X_y(
            X,
            y,
            accept_sparse=False,
            dtype=np.float64,
            multi_output=True,
            y_numeric=True,
        )
        self.n_features_in_ = int(X_checked.shape[1])
        X_array = np.array(X_checked, dtype=np.float64, copy=self.copy)
        y_array_raw = np.asarray(y_checked, dtype=np.float64)
        self._y_was_1d = y_array_raw.ndim == 1
        y_array = np.array(
            y_array_raw.reshape(-1, 1) if self._y_was_1d else y_array_raw,
            dtype=np.float64,
            copy=self.copy,
        )

        self._validate_constructor_parameters()
        self.n_targets_ = int(y_array.shape[1])
        if self.n_components > self.n_targets_:
            raise ValueError(
                "n_components must not exceed the number of response columns: "
                f"got n_components={self.n_components}, n_targets={self.n_targets_}."
            )
        self._clear_selection_attributes()

        if self.predictor_rank in ("auto", "optimal"):
            predictor_rank, max_predictor_rank = self._select_cross_validated_rank(
                X_array,
                y_array_raw,
                search_method=self.predictor_rank,
            )
        else:
            max_predictor_rank = _max_predictor_rank(
                n_features=self.n_features_in_,
                n_train_min=int(X_array.shape[0]),
                samples_per_predictor_rank=self.samples_per_predictor_rank,
            )
            predictor_rank = (
                max_predictor_rank if self.predictor_rank == "max" else int(self.predictor_rank)
            )

        if self.n_components > predictor_rank:
            raise ValueError(
                "n_components must satisfy n_components <= predictor_rank_; "
                f"got n_components={self.n_components}, predictor_rank_={predictor_rank}."
            )

        self._fit_fixed_rank(
            X_array,
            y_array,
            predictor_rank=predictor_rank,
            max_predictor_rank=max_predictor_rank,
        )
        return self

    def predict(self, X: ArrayLike) -> FloatArray:
        """Predict responses in their original units."""

        check_is_fitted(self, attributes=["coef_", "intercept_"])
        X_checked = _check_predictor_matrix(X, n_features=self.n_features_in_)
        prediction = cast(
            FloatArray,
            np.asarray(X_checked, dtype=np.float64) @ self.coef_.T + self.intercept_,
        )
        if self._y_was_1d:
            return prediction[:, 0]
        return prediction

    def transform(
        self,
        X: ArrayLike,
        y: ArrayLike | None = None,
    ) -> FloatArray | tuple[FloatArray, FloatArray]:
        """Transform predictors, and optionally responses, to latent scores."""

        check_is_fitted(self, attributes=["P_", "Q_", "x_mean_", "y_mean_"])
        X_checked = _check_predictor_matrix(X, n_features=self.n_features_in_)
        X_cs = (np.asarray(X_checked, dtype=np.float64) - self.x_mean_) / self.x_scale_
        x_scores = cast(FloatArray, X_cs @ self.P_)
        if y is None:
            return x_scores

        y_checked = check_array(
            y,
            ensure_2d=False,
            dtype=np.float64,
            ensure_min_samples=X_cs.shape[0],
        )
        y_array = np.asarray(y_checked, dtype=np.float64)
        if y_array.ndim == 1:
            y_array = y_array.reshape(-1, 1)
        if y_array.shape[0] != X_cs.shape[0]:
            raise ValueError(
                "X and y must contain the same number of samples: "
                f"got {X_cs.shape[0]} and {y_array.shape[0]}."
            )
        if y_array.shape[1] != self.n_targets_:
            raise ValueError(
                "y has an incompatible number of targets: "
                f"expected {self.n_targets_}, got {y_array.shape[1]}."
            )
        y_cs = (y_array - self.y_mean_) / self.y_scale_
        y_scores = cast(FloatArray, y_cs @ self.Q_)
        return x_scores, y_scores

    def score(self, X: ArrayLike, y: ArrayLike, sample_weight: ArrayLike | None = None) -> float:
        """Return uniformly averaged $R^2$ in original response units."""

        return float(
            r2_score(
                y,
                self.predict(X),
                sample_weight=sample_weight,
                multioutput="uniform_average",
            )
        )

    def _select_cross_validated_rank(
        self,
        X: FloatArray,
        y: FloatArray,
        *,
        search_method: Literal["auto", "optimal"],
    ) -> tuple[int, int]:
        materialized = _materialize_cv_splits(self.cv, X, y)
        max_predictor_rank = _max_predictor_rank(
            n_features=self.n_features_in_,
            n_train_min=materialized.n_train_min,
            samples_per_predictor_rank=self.samples_per_predictor_rank,
        )
        admissible_ranks = _predictor_rank_values(
            n_components=self.n_components,
            max_predictor_rank=max_predictor_rank,
        )
        scorer = _resolve_scorer(self.scoring)
        cache: dict[int, _CandidateCVResult] = {}
        evaluation_order: list[int] = []
        search_history: list[IntArray] = []

        def evaluate(ranks: IntArray) -> None:
            pending = np.asarray(
                sorted({int(rank) for rank in ranks if int(rank) not in cache}),
                dtype=np.intp,
            )
            if pending.size == 0:
                return
            evaluations = _evaluate_predictor_ranks(
                predictor_ranks=pending,
                n_components=self.n_components,
                scale=bool(self.scale),
                copy=bool(self.copy),
                samples_per_predictor_rank=self.samples_per_predictor_rank,
                scorer=scorer,
                X=X,
                y=y,
                splits=materialized.splits,
                n_jobs=self.n_jobs,
            )
            for result in evaluations:
                cache[result.predictor_rank] = result
                evaluation_order.append(result.predictor_rank)
            search_history.append(pending.copy())

        lower = int(admissible_ranks[0])
        upper = int(admissible_ranks[-1])
        if search_method == "optimal" or admissible_ranks.size <= _ADAPTIVE_EXHAUSTIVE_THRESHOLD:
            evaluate(admissible_ranks)
            final_interval = (lower, upper)
        else:
            evaluate(
                _logarithmic_predictor_rank_values(
                    lower=lower,
                    upper=upper,
                )
            )
            while True:
                evaluated_ranks, mean_scores = _cached_mean_scores(cache)
                interval_lower, interval_upper = _adaptive_refinement_interval(
                    evaluated_ranks,
                    -mean_scores,
                )
                interval_values = np.arange(
                    interval_lower,
                    interval_upper + 1,
                    dtype=np.intp,
                )
                missing = np.asarray(
                    [rank for rank in interval_values if int(rank) not in cache],
                    dtype=np.intp,
                )
                if interval_values.size <= _ADAPTIVE_EXHAUSTIVE_THRESHOLD:
                    evaluate(missing)
                    final_interval = (interval_lower, interval_upper)
                    break

                proposed = _logarithmic_predictor_rank_values(
                    lower=interval_lower,
                    upper=interval_upper,
                )
                missing = np.asarray(
                    [rank for rank in proposed if int(rank) not in cache],
                    dtype=np.intp,
                )
                if missing.size == 0:
                    unevaluated = np.asarray(
                        [rank for rank in interval_values if int(rank) not in cache],
                        dtype=np.intp,
                    )
                    if unevaluated.size == 0:
                        final_interval = (interval_lower, interval_upper)
                        break
                    missing = unevaluated[unevaluated.size // 2 : unevaluated.size // 2 + 1]
                evaluate(missing)

        predictor_rank_values = np.asarray(sorted(cache), dtype=np.intp)
        evaluations = [cache[int(rank)] for rank in predictor_rank_values]
        split_scores = np.vstack([result.split_scores for result in evaluations])
        split_mse = np.vstack([result.split_response_standardized_mse for result in evaluations])
        mean_scores = np.mean(split_scores, axis=1)
        mean_mse = np.mean(split_mse, axis=1)
        selected_rank, _ = _select_predictor_rank(
            predictor_rank_values,
            -mean_scores,
        )
        selected_index = int(np.flatnonzero(predictor_rank_values == selected_rank)[0])

        self.predictor_rank_values_ = predictor_rank_values.copy()
        self.predictor_rank_evaluation_order_ = np.asarray(
            evaluation_order,
            dtype=np.intp,
        )
        self.predictor_rank_search_history_ = tuple(batch.copy() for batch in search_history)
        self.predictor_rank_cv_results_ = _cv_results_dictionary(
            predictor_rank_values=predictor_rank_values,
            split_scores=split_scores,
            split_response_standardized_mse=split_mse,
        )
        self.predictor_rank_search_method_ = search_method
        self.predictor_rank_search_interval_ = np.asarray(
            final_interval,
            dtype=np.intp,
        )
        self.n_predictor_rank_candidates_ = int(admissible_ranks.size)
        self.n_predictor_rank_evaluated_ = int(predictor_rank_values.size)
        self.n_predictor_rank_skipped_ = (
            self.n_predictor_rank_candidates_ - self.n_predictor_rank_evaluated_
        )
        self.predictor_rank_search_exhaustive_ = (
            self.n_predictor_rank_evaluated_ == self.n_predictor_rank_candidates_
        )
        self.best_score_ = float(mean_scores[selected_index])
        self.best_response_standardized_mse_ = float(mean_mse[selected_index])
        self.n_splits_ = len(materialized.splits)
        self.cv_n_train_min_ = materialized.n_train_min
        return selected_rank, max_predictor_rank

    def _fit_fixed_rank(
        self,
        X: FloatArray,
        y: FloatArray,
        *,
        predictor_rank: int,
        max_predictor_rank: int,
    ) -> None:
        self.x_mean_ = np.mean(X, axis=0)
        self.y_mean_ = np.mean(y, axis=0)
        X_centered = X - self.x_mean_
        y_centered = y - self.y_mean_

        if self.scale:
            self.x_scale_ = _safe_sample_scale(X_centered)
            self.y_scale_ = _safe_sample_scale(y_centered)
        else:
            self.x_scale_ = np.ones(X.shape[1], dtype=np.float64)
            self.y_scale_ = np.ones(y.shape[1], dtype=np.float64)
        self.response_scale_for_scoring_ = _training_response_scale(y)

        X_cs = X_centered / self.x_scale_
        y_cs = y_centered / self.y_scale_
        result = fit_pipls_core(
            X_cs,
            y_cs,
            predictor_rank=predictor_rank,
            n_components=self.n_components,
        )

        self.predictor_rank_ = predictor_rank
        self.max_predictor_rank_ = max_predictor_rank
        self.Pi_ = result.Pi
        self.C_ = result.C
        self.W_ = result.W
        self.P_ = result.P
        self.D_ = result.D
        self.dilation_ = np.diag(result.D).copy()
        self.Q_ = result.Q
        self.x_rotations_ = self.P_
        self.y_rotations_ = self.Q_
        self.x_rank_ = result.x_rank
        self.rank_tolerance_ = result.rank_tolerance

        self.coef_matrix_ = result.regression_map * self.y_scale_[None, :] / self.x_scale_[:, None]
        self.coef_ = self.coef_matrix_.T
        self.intercept_ = self.y_mean_ - self.x_mean_ @ self.coef_matrix_
        self.x_scores_ = X_cs @ self.P_
        self.y_scores_ = y_cs @ self.Q_

    def _validate_constructor_parameters(self) -> None:
        _validate_positive_int(self.n_components, name="n_components")
        if self.predictor_rank in ("max", "optimal", "auto"):
            pass
        elif isinstance(self.predictor_rank, (int, np.integer)) and not isinstance(
            self.predictor_rank,
            (bool, np.bool_),
        ):
            _validate_positive_int(self.predictor_rank, name="predictor_rank")
            if self.n_components > self.predictor_rank:
                raise ValueError(
                    "n_components must satisfy n_components <= predictor_rank; "
                    f"got n_components={self.n_components}, "
                    f"predictor_rank={self.predictor_rank}."
                )
        else:
            raise ValueError(
                'predictor_rank must be a positive integer, "max", "optimal", or "auto"; '
                f"got {self.predictor_rank!r}."
            )
        _as_positive_float(
            self.samples_per_predictor_rank,
            name="samples_per_predictor_rank",
        )
        if not isinstance(self.scale, (bool, np.bool_)):
            raise ValueError(f"scale must be boolean; got {self.scale!r}.")
        if not isinstance(self.copy, (bool, np.bool_)):
            raise ValueError(f"copy must be boolean; got {self.copy!r}.")
        if self.predictor_rank in ("auto", "optimal"):
            _validate_n_jobs(self.n_jobs)
            _resolve_scorer(self.scoring)

    def _clear_selection_attributes(self) -> None:
        for name in (
            "predictor_rank_values_",
            "predictor_rank_cv_results_",
            "best_score_",
            "best_response_standardized_mse_",
            "n_splits_",
            "cv_n_train_min_",
            "predictor_rank_evaluation_order_",
            "predictor_rank_search_history_",
            "predictor_rank_search_method_",
            "predictor_rank_search_interval_",
            "n_predictor_rank_candidates_",
            "n_predictor_rank_evaluated_",
            "n_predictor_rank_skipped_",
            "predictor_rank_search_exhaustive_",
        ):
            if hasattr(self, name):
                delattr(self, name)


def _evaluate_predictor_ranks(
    *,
    predictor_ranks: IntArray,
    n_components: int,
    scale: bool,
    copy: bool,
    samples_per_predictor_rank: float,
    scorer: Scorer,
    X: FloatArray,
    y: FloatArray,
    splits: tuple[tuple[IntArray, IntArray], ...],
    n_jobs: int | None,
) -> list[_CandidateCVResult]:
    return cast(
        list[_CandidateCVResult],
        Parallel(n_jobs=n_jobs, prefer="threads")(
            delayed(_evaluate_predictor_rank)(
                predictor_rank=int(predictor_rank),
                n_components=n_components,
                scale=scale,
                copy=copy,
                samples_per_predictor_rank=samples_per_predictor_rank,
                scorer=scorer,
                X=X,
                y=y,
                splits=splits,
            )
            for predictor_rank in predictor_ranks
        ),
    )


def _cached_mean_scores(
    cache: dict[int, _CandidateCVResult],
) -> tuple[IntArray, FloatArray]:
    ranks = np.asarray(sorted(cache), dtype=np.intp)
    scores = np.asarray(
        [np.mean(cache[int(rank)].split_scores) for rank in ranks],
        dtype=np.float64,
    )
    return ranks, scores


def _evaluate_predictor_rank(
    *,
    predictor_rank: int,
    n_components: int,
    scale: bool,
    copy: bool,
    samples_per_predictor_rank: float,
    scorer: Scorer,
    X: FloatArray,
    y: FloatArray,
    splits: tuple[tuple[IntArray, IntArray], ...],
) -> _CandidateCVResult:
    split_scores = np.empty(len(splits), dtype=np.float64)
    split_mse = np.empty(len(splits), dtype=np.float64)
    for split_index, (train, validation) in enumerate(splits):
        model = PiPLSRegression(
            n_components=n_components,
            scale=scale,
            copy=copy,
            predictor_rank=predictor_rank,
            samples_per_predictor_rank=samples_per_predictor_rank,
        )
        y_train = y[train]
        model.fit(X[train], y_train)
        y_prediction = model.predict(X[validation])
        score = float(scorer(model, X[validation], y[validation]))
        if not np.isfinite(score):
            raise ValueError(
                "The scoring callable returned a nonfinite value for "
                f"predictor_rank={predictor_rank}, split={split_index}."
            )
        split_scores[split_index] = score
        split_mse[split_index] = _response_standardized_mse(
            y[validation],
            y_prediction,
            _training_response_scale(y_train),
        )
    return _CandidateCVResult(
        predictor_rank=predictor_rank,
        split_scores=split_scores,
        split_response_standardized_mse=split_mse,
    )


def _cv_results_dictionary(
    *,
    predictor_rank_values: NDArray[np.intp],
    split_scores: FloatArray,
    split_response_standardized_mse: FloatArray,
) -> dict[str, FloatArray | NDArray[np.intp]]:
    results: dict[str, FloatArray | NDArray[np.intp]] = {
        "predictor_rank": predictor_rank_values.copy(),
        "mean_test_score": np.mean(split_scores, axis=1),
        "std_test_score": np.std(split_scores, axis=1),
        "mean_response_standardized_mse": np.mean(
            split_response_standardized_mse,
            axis=1,
        ),
        "std_response_standardized_mse": np.std(
            split_response_standardized_mse,
            axis=1,
        ),
    }
    for split_index in range(split_scores.shape[1]):
        results[f"split{split_index}_test_score"] = split_scores[:, split_index].copy()
        results[f"split{split_index}_response_standardized_mse"] = split_response_standardized_mse[
            :, split_index
        ].copy()
    return results


def _resolve_scorer(scoring: str | Scorer) -> Scorer:
    if scoring == _DEFAULT_SCORING:
        return neg_response_standardized_mean_squared_error
    if isinstance(scoring, str):
        try:
            return cast(Scorer, get_scorer(scoring))
        except ValueError as error:
            raise ValueError(f"Unknown scoring value {scoring!r}.") from error
    if callable(scoring):
        return scoring
    raise ValueError(f"scoring must be a scikit-learn scorer name or callable; got {scoring!r}.")


def _validate_n_jobs(n_jobs: int | None) -> None:
    if n_jobs is None:
        return
    if isinstance(n_jobs, (bool, np.bool_)) or not isinstance(n_jobs, (int, np.integer)):
        raise ValueError(f"n_jobs must be None or a nonzero integer; got {n_jobs!r}.")
    if int(n_jobs) == 0:
        raise ValueError("n_jobs must not be zero.")


def _check_predictor_matrix(X: ArrayLike, *, n_features: int) -> FloatArray:
    checked = check_array(X, accept_sparse=False, dtype=np.float64)
    array = np.asarray(checked, dtype=np.float64)
    if array.shape[1] != n_features:
        raise ValueError(
            "X has an incompatible number of features: "
            f"expected {n_features}, got {array.shape[1]}."
        )
    return array


def _safe_sample_scale(centered: FloatArray) -> FloatArray:
    if centered.shape[0] <= 1:
        return np.ones(centered.shape[1], dtype=np.float64)
    scale = np.std(centered, axis=0, ddof=1)
    return np.where(scale == 0.0, 1.0, scale).astype(np.float64, copy=False)


def _validate_positive_int(value: Any, *, name: str) -> None:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be a positive integer; got {value!r}.")
    if int(value) < 1:
        raise ValueError(f"{name} must be at least 1; got {value!r}.")
