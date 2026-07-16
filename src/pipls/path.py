"""Pipeline-aware cross-validated Pi-PLS path analysis."""

from __future__ import annotations

import warnings
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Any, Literal, cast

import numpy as np
from joblib import Parallel, delayed
from numpy.typing import ArrayLike, NDArray
from sklearn.base import BaseEstimator, clone
from sklearn.metrics import get_scorer
from sklearn.utils import _safe_indexing
from sklearn.utils.validation import check_is_fitted, check_X_y

from .exceptions import StatisticalSupportWarning
from .model_selection import (
    _ADAPTIVE_EXHAUSTIVE_THRESHOLD,
    _as_positive_float,
    _logarithmic_predictor_rank_values,
    _materialize_cv_splits,
    _max_predictor_rank,
    _response_standardized_mse,
    _training_response_scale,
    _validate_positive_int,
)
from .regression import PiPLSRegression

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.intp]
Scorer = Callable[[Any, ArrayLike, ArrayLike], float]
SearchMethod = Literal["optimal", "auto"]
_DEFAULT_SCORING = "neg_response_standardized_mean_squared_error"
_MIN_TRUSTED_SAMPLES_PER_PREDICTOR_RANK = 5.0
_SELECTION_RTOL = 1e-12
_SELECTION_ATOL = 1e-15


@dataclass(frozen=True)
class _PathCandidateResult:
    """Cross-validation result for one admissible ``(h, r_pi)`` pair."""

    n_components: int
    predictor_rank: int
    split_scores: FloatArray
    split_response_standardized_mse: FloatArray


class PiPLSPathCV(BaseEstimator):  # type: ignore[misc]
    r"""Cross-validated search over the admissible Pi-PLS rank path.

    The default ``search_method="auto"`` evaluates the complete triangular
    grid. ``search_method="auto"`` applies the same deterministic logarithmic
    coarse-to-fine predictor-rank search used by :class:`PiPLSRegression`
    independently for each value of ``n_components``.

    Parameters
    ----------
    estimator:
        Estimator or composite estimator containing exactly one
        :class:`PiPLSRegression`. ``None`` creates a default estimator template.
    pipls_param_prefix:
        Nested parameter prefix locating the Pi-PLS estimator, for example
        ``"regression"`` or ``"regressor__regression"``. It is inferred when
        exactly one nested Pi-PLS estimator is present.
    n_components_values:
        Positive component counts to evaluate. ``None`` uses every value from 1
        through ``min(n_targets, max_predictor_rank_)``.
    predictor_rank_values:
        Positive predictor ranks to evaluate. ``None`` uses every value from 1
        through ``max_predictor_rank_``.
    max_predictor_rank:
        ``"rule"`` uses the fold-safe rank rule. A positive integer imposes an
        additional explicit upper bound.
    search_method:
        ``"optimal"`` evaluates every admissible pair; ``"auto"`` is adaptive
        and approximate.
    samples_per_predictor_rank:
        Positive rank-bound parameter $c$. Values below 5 issue
        :class:`StatisticalSupportWarning`.
    cv:
        Integer split count, splitter, or iterable of train-validation pairs.
    scoring:
        Scikit-learn scorer name or callable. The default is negative
        response-standardized MSE.
    refit:
        Refit the globally selected pair on all supplied data.
    n_jobs:
        Joblib parallelism across candidate pairs within each evaluation batch.
    """

    def __init__(
        self,
        estimator: Any | None = None,
        *,
        pipls_param_prefix: str | None = None,
        n_components_values: Sequence[int] | None = None,
        predictor_rank_values: Sequence[int] | None = None,
        max_predictor_rank: int | Literal["rule"] = "rule",
        search_method: SearchMethod = "auto",
        samples_per_predictor_rank: float = 10.0,
        cv: object = 5,
        scoring: str | Scorer = _DEFAULT_SCORING,
        refit: bool = True,
        n_jobs: int | None = None,
    ) -> None:
        self.estimator = estimator
        self.pipls_param_prefix = pipls_param_prefix
        self.n_components_values = n_components_values
        self.predictor_rank_values = predictor_rank_values
        self.max_predictor_rank = max_predictor_rank
        self.search_method = search_method
        self.samples_per_predictor_rank = samples_per_predictor_rank
        self.cv = cv
        self.scoring = scoring
        self.refit = refit
        self.n_jobs = n_jobs

    def fit(
        self,
        X: ArrayLike,
        y: ArrayLike,
        *,
        groups: ArrayLike | None = None,
    ) -> PiPLSPathCV:
        """Evaluate the path and optionally refit the globally selected pair."""

        self._validate_constructor_parameters()
        X_checked, y_checked = check_X_y(
            X,
            y,
            accept_sparse=False,
            dtype=np.float64,
            multi_output=True,
            y_numeric=True,
        )
        X_array = np.asarray(X_checked, dtype=np.float64)
        y_array = np.asarray(y_checked, dtype=np.float64)
        y_2d = y_array.reshape(-1, 1) if y_array.ndim == 1 else y_array
        self.n_features_in_ = int(X_array.shape[1])
        self.n_targets_ = int(y_2d.shape[1])

        template = PiPLSRegression() if self.estimator is None else self.estimator
        self.pipls_param_prefix_ = _resolve_pipls_param_prefix(
            template,
            self.pipls_param_prefix,
        )
        n_components_key, predictor_rank_key = _pipls_parameter_keys(
            self.pipls_param_prefix_
        )
        materialized = _materialize_cv_splits(self.cv, X_array, y_array, groups=groups)
        self.n_splits_ = len(materialized.splits)
        self.cv_n_train_min_ = materialized.n_train_min

        fold_feature_limit = _fold_safe_feature_limit(
            template=template,
            prefix=self.pipls_param_prefix_,
            X=X_array,
            y=y_array,
            splits=materialized.splits,
        )
        algebraic_limit = min(fold_feature_limit, materialized.n_train_min)
        if self.max_predictor_rank == "rule":
            self.max_predictor_rank_ = _max_predictor_rank(
                n_features=fold_feature_limit,
                n_train_min=materialized.n_train_min,
                samples_per_predictor_rank=self.samples_per_predictor_rank,
            )
        else:
            self.max_predictor_rank_ = min(int(self.max_predictor_rank), algebraic_limit)

        h_limit = min(self.n_targets_, self.max_predictor_rank_)
        self.n_components_values_ = _validated_integer_values(
            self.n_components_values,
            name="n_components_values",
            lower=1,
            upper=h_limit,
        )
        self.predictor_rank_values_ = _validated_integer_values(
            self.predictor_rank_values,
            name="predictor_rank_values",
            lower=1,
            upper=self.max_predictor_rank_,
        )
        admissible = tuple(
            (int(h), int(r))
            for h in self.n_components_values_
            for r in self.predictor_rank_values_
            if h <= r
        )
        if not admissible:
            raise ValueError(
                "The supplied n_components_values and predictor_rank_values do not "
                "contain an admissible pair satisfying n_components <= predictor_rank."
            )
        self.n_path_candidates_ = len(admissible)

        scorer = _resolve_path_scorer(self.scoring)
        cache: dict[tuple[int, int], _PathCandidateResult] = {}
        history: dict[int, tuple[tuple[int, ...], ...]] = {}
        if self.search_method == "optimal":
            _evaluate_path_batch(
                pairs=admissible,
                cache=cache,
                template=template,
                n_components_key=n_components_key,
                predictor_rank_key=predictor_rank_key,
                scorer=scorer,
                scoring=self.scoring,
                X=X_array,
                y=y_array,
                splits=materialized.splits,
                n_jobs=self.n_jobs,
            )
            for h in self.n_components_values_:
                history[int(h)] = (
                    tuple(r for hh, r in admissible if hh == int(h)),
                )
        else:
            for h_value in self.n_components_values_:
                h = int(h_value)
                allowed = np.asarray(
                    [r for hh, r in admissible if hh == h],
                    dtype=np.intp,
                )
                history[h] = _adaptive_path_search(
                    n_components=h,
                    allowed_ranks=allowed,
                    cache=cache,
                    template=template,
                    n_components_key=n_components_key,
                    predictor_rank_key=predictor_rank_key,
                    scorer=scorer,
                    scoring=self.scoring,
                    X=X_array,
                    y=y_array,
                    splits=materialized.splits,
                    n_jobs=self.n_jobs,
                )

        evaluated_pairs = tuple(sorted(cache))
        self.n_path_candidates_evaluated_ = len(evaluated_pairs)
        self.n_path_candidates_skipped_ = (
            self.n_path_candidates_ - self.n_path_candidates_evaluated_
        )
        self.path_search_exhaustive_ = self.n_path_candidates_skipped_ == 0
        self.path_search_method_ = self.search_method
        self.path_search_history_ = history

        self.cv_results_ = _path_cv_results(
            cache=cache,
            evaluated_pairs=evaluated_pairs,
            n_components_key=n_components_key,
            predictor_rank_key=predictor_rank_key,
        )
        self.best_index_ = _select_global_best_index(self.cv_results_)
        self.best_score_ = float(self.cv_results_["mean_test_score"][self.best_index_])
        self.best_response_standardized_mse_ = float(
            self.cv_results_["mean_response_standardized_mse"][self.best_index_]
        )
        self.best_n_components_ = int(
            self.cv_results_["n_components"][self.best_index_]
        )
        self.best_predictor_rank_ = int(
            self.cv_results_["predictor_rank"][self.best_index_]
        )
        self.best_params_ = {
            n_components_key: self.best_n_components_,
            predictor_rank_key: self.best_predictor_rank_,
        }

        self.best_predictor_rank_by_n_components_ = {}
        self.best_score_by_n_components_ = {}
        for h_value in self.n_components_values_:
            h = int(h_value)
            indices = np.flatnonzero(self.cv_results_["n_components"] == h)
            if indices.size == 0:
                continue
            local_index = _select_conditional_best_index(self.cv_results_, indices)
            self.best_predictor_rank_by_n_components_[h] = int(
                self.cv_results_["predictor_rank"][local_index]
            )
            self.best_score_by_n_components_[h] = float(
                self.cv_results_["mean_test_score"][local_index]
            )

        self.response_standardized_mse_path_ = _path_surface(
            self.n_components_values_,
            self.predictor_rank_values_,
            self.cv_results_["n_components"],
            self.cv_results_["predictor_rank"],
            self.cv_results_["mean_response_standardized_mse"],
        )
        self.score_path_ = _path_surface(
            self.n_components_values_,
            self.predictor_rank_values_,
            self.cv_results_["n_components"],
            self.cv_results_["predictor_rank"],
            self.cv_results_["mean_test_score"],
        )

        if self.refit:
            self.best_estimator_ = clone(template).set_params(**self.best_params_)
            self.best_estimator_.fit(X_array, y_array)
        elif hasattr(self, "best_estimator_"):
            delattr(self, "best_estimator_")
        return self

    def predict(self, X: ArrayLike) -> FloatArray:
        """Predict with the refitted globally selected estimator."""

        estimator = self._refitted_estimator()
        return cast(FloatArray, estimator.predict(X))

    def transform(self, X: ArrayLike, y: ArrayLike | None = None) -> Any:
        """Transform with the refitted globally selected estimator."""

        estimator = self._refitted_estimator()
        if y is None:
            return estimator.transform(X)
        return estimator.transform(X, y)

    def score(self, X: ArrayLike, y: ArrayLike) -> float:
        """Score with the refitted globally selected estimator."""

        estimator = self._refitted_estimator()
        return float(estimator.score(X, y))

    def _refitted_estimator(self) -> Any:
        check_is_fitted(self, attributes=["best_params_"])
        if not hasattr(self, "best_estimator_"):
            raise AttributeError(
                "PiPLSPathCV was fitted with refit=False; predict, transform, and score "
                "require refit=True."
            )
        return self.best_estimator_

    def _validate_constructor_parameters(self) -> None:
        if self.estimator is not None and not hasattr(self.estimator, "get_params"):
            raise ValueError("estimator must implement the scikit-learn estimator interface.")
        if self.pipls_param_prefix is not None and not isinstance(
            self.pipls_param_prefix, str
        ):
            raise ValueError("pipls_param_prefix must be None or a string.")
        if self.search_method not in ("optimal", "auto"):
            raise ValueError('search_method must be "optimal" or "auto".')
        if not isinstance(self.refit, (bool, np.bool_)):
            raise ValueError(f"refit must be boolean; got {self.refit!r}.")
        samples_per_rank = _as_positive_float(
            self.samples_per_predictor_rank,
            name="samples_per_predictor_rank",
        )
        if (
            self.max_predictor_rank == "rule"
            and samples_per_rank < _MIN_TRUSTED_SAMPLES_PER_PREDICTOR_RANK
        ):
            warnings.warn(
                f"samples_per_predictor_rank={samples_per_rank:g} is below 5. "
                "This permits fewer than five training samples per retained predictor-rank "
                "direction, so the resulting rank bound may not have sufficient statistical "
                "support to be trusted without external validation.",
                StatisticalSupportWarning,
                stacklevel=3,
            )
        if self.max_predictor_rank != "rule":
            _validate_positive_int(self.max_predictor_rank, name="max_predictor_rank")
        _validate_optional_integer_sequence(
            self.n_components_values,
            name="n_components_values",
        )
        _validate_optional_integer_sequence(
            self.predictor_rank_values,
            name="predictor_rank_values",
        )
        _validate_n_jobs(self.n_jobs)
        _validate_cv(self.cv)
        _resolve_path_scorer(self.scoring)


def _resolve_pipls_param_prefix(template: Any, supplied: str | None) -> str:
    params = template.get_params(deep=True)
    if supplied is not None:
        prefix = supplied.removesuffix("__")
        n_key, r_key = _pipls_parameter_keys(prefix)
        if n_key not in params or r_key not in params:
            raise ValueError(
                f"pipls_param_prefix={supplied!r} does not locate n_components and "
                "predictor_rank parameters in estimator."
            )
        return prefix
    if isinstance(template, PiPLSRegression):
        return ""
    matches = sorted(
        key for key, value in params.items() if isinstance(value, PiPLSRegression)
    )
    if len(matches) != 1:
        raise ValueError(
            "Could not infer a unique PiPLSRegression parameter location; supply "
            "pipls_param_prefix explicitly."
        )
    return cast(str, matches[0])


def _pipls_parameter_keys(prefix: str) -> tuple[str, str]:
    separator = "__" if prefix else ""
    return (
        f"{prefix}{separator}n_components",
        f"{prefix}{separator}predictor_rank",
    )


def _extract_fitted_pipls(estimator: Any, prefix: str) -> PiPLSRegression:
    if prefix == "":
        if not isinstance(estimator, PiPLSRegression):
            raise ValueError("The resolved estimator is not PiPLSRegression.")
        return estimator
    value = estimator.get_params(deep=True).get(prefix)
    if not isinstance(value, PiPLSRegression):
        raise ValueError(f"Fitted parameter prefix {prefix!r} no longer locates PiPLSRegression.")
    return value


def _fold_safe_feature_limit(
    *,
    template: Any,
    prefix: str,
    X: FloatArray,
    y: FloatArray,
    splits: tuple[tuple[IntArray, IntArray], ...],
) -> int:
    if prefix == "" and isinstance(template, PiPLSRegression):
        return int(X.shape[1])
    n_key, r_key = _pipls_parameter_keys(prefix)
    feature_counts: list[int] = []
    for train, _ in splits:
        probe = clone(template).set_params(**{n_key: 1, r_key: 1})
        probe.fit(_safe_indexing(X, train), _safe_indexing(y, train))
        feature_counts.append(_extract_fitted_pipls(probe, prefix).n_features_in_)
    return min(feature_counts)


def _evaluate_path_batch(
    *,
    pairs: Iterable[tuple[int, int]],
    cache: dict[tuple[int, int], _PathCandidateResult],
    template: Any,
    n_components_key: str,
    predictor_rank_key: str,
    scorer: Scorer | None,
    scoring: str | Scorer,
    X: FloatArray,
    y: FloatArray,
    splits: tuple[tuple[IntArray, IntArray], ...],
    n_jobs: int | None,
) -> tuple[int, ...]:
    new_pairs = tuple(pair for pair in pairs if pair not in cache)
    if not new_pairs:
        return ()
    results = Parallel(n_jobs=n_jobs)(
        delayed(_evaluate_path_candidate)(
            n_components=h,
            predictor_rank=r,
            template=template,
            n_components_key=n_components_key,
            predictor_rank_key=predictor_rank_key,
            scorer=scorer,
            scoring=scoring,
            X=X,
            y=y,
            splits=splits,
        )
        for h, r in new_pairs
    )
    for result in results:
        cache[(result.n_components, result.predictor_rank)] = result
    return tuple(r for _, r in new_pairs)


def _evaluate_path_candidate(
    *,
    n_components: int,
    predictor_rank: int,
    template: Any,
    n_components_key: str,
    predictor_rank_key: str,
    scorer: Scorer | None,
    scoring: str | Scorer,
    X: FloatArray,
    y: FloatArray,
    splits: tuple[tuple[IntArray, IntArray], ...],
) -> _PathCandidateResult:
    split_scores = np.empty(len(splits), dtype=np.float64)
    split_mse = np.empty(len(splits), dtype=np.float64)
    params = {
        n_components_key: n_components,
        predictor_rank_key: predictor_rank,
    }
    for split_index, (train, validation) in enumerate(splits):
        estimator = clone(template).set_params(**params)
        X_train = _safe_indexing(X, train)
        y_train = _safe_indexing(y, train)
        X_validation = _safe_indexing(X, validation)
        y_validation = _safe_indexing(y, validation)
        estimator.fit(X_train, y_train)
        prediction = estimator.predict(X_validation)
        mse = _response_standardized_mse(
            y_validation,
            prediction,
            _training_response_scale(y_train),
        )
        split_mse[split_index] = mse
        if scoring == _DEFAULT_SCORING:
            split_scores[split_index] = -mse
        else:
            assert scorer is not None
            score = float(scorer(estimator, X_validation, y_validation))
            if not np.isfinite(score):
                raise ValueError(
                    "The scoring callable returned a nonfinite value for "
                    f"n_components={n_components}, predictor_rank={predictor_rank}, "
                    f"split={split_index}."
                )
            split_scores[split_index] = score
    return _PathCandidateResult(
        n_components=n_components,
        predictor_rank=predictor_rank,
        split_scores=split_scores,
        split_response_standardized_mse=split_mse,
    )


def _adaptive_path_search(
    *,
    n_components: int,
    allowed_ranks: IntArray,
    cache: dict[tuple[int, int], _PathCandidateResult],
    template: Any,
    n_components_key: str,
    predictor_rank_key: str,
    scorer: Scorer | None,
    scoring: str | Scorer,
    X: FloatArray,
    y: FloatArray,
    splits: tuple[tuple[IntArray, IntArray], ...],
    n_jobs: int | None,
) -> tuple[tuple[int, ...], ...]:
    history: list[tuple[int, ...]] = []
    interval = allowed_ranks
    while True:
        if interval.size <= _ADAPTIVE_EXHAUSTIVE_THRESHOLD:
            ranks = interval
        else:
            proposed = _logarithmic_predictor_rank_values(
                lower=int(interval[0]),
                upper=int(interval[-1]),
            )
            indices = np.abs(interval[:, None] - proposed[None, :]).argmin(axis=0)
            ranks = np.unique(interval[indices])
        evaluated = _evaluate_path_batch(
            pairs=((n_components, int(rank)) for rank in ranks),
            cache=cache,
            template=template,
            n_components_key=n_components_key,
            predictor_rank_key=predictor_rank_key,
            scorer=scorer,
            scoring=scoring,
            X=X,
            y=y,
            splits=splits,
            n_jobs=n_jobs,
        )
        if evaluated:
            history.append(evaluated)
        if interval.size <= _ADAPTIVE_EXHAUSTIVE_THRESHOLD:
            break
        evaluated_ranks = np.asarray(
            sorted(r for h, r in cache if h == n_components),
            dtype=np.intp,
        )
        scores = np.asarray(
            [
                np.mean(cache[(n_components, int(rank))].split_scores)
                for rank in evaluated_ranks
            ],
            dtype=np.float64,
        )
        best_rank = _best_rank(evaluated_ranks, scores)
        best_index = int(np.flatnonzero(evaluated_ranks == best_rank)[0])
        lower = int(evaluated_ranks[max(0, best_index - 1)])
        upper = int(evaluated_ranks[min(evaluated_ranks.size - 1, best_index + 1)])
        refined = allowed_ranks[(allowed_ranks >= lower) & (allowed_ranks <= upper)]
        if np.array_equal(refined, interval):
            remaining = np.asarray(
                [r for r in interval if (n_components, int(r)) not in cache],
                dtype=np.intp,
            )
            if remaining.size:
                _evaluate_path_batch(
                    pairs=((n_components, int(rank)) for rank in remaining),
                    cache=cache,
                    template=template,
                    n_components_key=n_components_key,
                    predictor_rank_key=predictor_rank_key,
                    scorer=scorer,
                    scoring=scoring,
                    X=X,
                    y=y,
                    splits=splits,
                    n_jobs=n_jobs,
                )
                history.append(tuple(int(rank) for rank in remaining))
            break
        interval = refined
    return tuple(history)


def _best_rank(ranks: IntArray, scores: FloatArray) -> int:
    maximum = float(np.max(scores))
    tied = np.isclose(scores, maximum, rtol=_SELECTION_RTOL, atol=_SELECTION_ATOL)
    return int(np.min(ranks[tied]))


def _path_cv_results(
    *,
    cache: dict[tuple[int, int], _PathCandidateResult],
    evaluated_pairs: tuple[tuple[int, int], ...],
    n_components_key: str,
    predictor_rank_key: str,
) -> dict[str, Any]:
    n_components = np.asarray([pair[0] for pair in evaluated_pairs], dtype=np.intp)
    predictor_rank = np.asarray([pair[1] for pair in evaluated_pairs], dtype=np.intp)
    split_scores = np.vstack([cache[pair].split_scores for pair in evaluated_pairs])
    split_mse = np.vstack(
        [cache[pair].split_response_standardized_mse for pair in evaluated_pairs]
    )
    mean_scores = np.mean(split_scores, axis=1)
    results: dict[str, Any] = {
        "params": [
            {n_components_key: int(h), predictor_rank_key: int(r)}
            for h, r in evaluated_pairs
        ],
        "n_components": n_components,
        "predictor_rank": predictor_rank,
        f"param_{n_components_key}": n_components.copy(),
        f"param_{predictor_rank_key}": predictor_rank.copy(),
        "mean_test_score": mean_scores,
        "std_test_score": np.std(split_scores, axis=1),
        "mean_response_standardized_mse": np.mean(split_mse, axis=1),
        "std_response_standardized_mse": np.std(split_mse, axis=1),
    }
    order = np.lexsort((predictor_rank, n_components, -mean_scores))
    rank_values = np.empty(order.size, dtype=np.intp)
    rank_values[order] = np.arange(1, order.size + 1, dtype=np.intp)
    results["rank_test_score"] = rank_values
    for split_index in range(split_scores.shape[1]):
        results[f"split{split_index}_test_score"] = split_scores[:, split_index].copy()
        results[f"split{split_index}_response_standardized_mse"] = split_mse[
            :, split_index
        ].copy()
    return results


def _select_global_best_index(results: dict[str, Any]) -> int:
    scores = cast(FloatArray, results["mean_test_score"])
    maximum = float(np.max(scores))
    tied = np.flatnonzero(
        np.isclose(scores, maximum, rtol=_SELECTION_RTOL, atol=_SELECTION_ATOL)
    )
    h = cast(IntArray, results["n_components"])[tied]
    r = cast(IntArray, results["predictor_rank"])[tied]
    return int(tied[np.lexsort((r, h))[0]])


def _select_conditional_best_index(results: dict[str, Any], indices: IntArray) -> int:
    scores = cast(FloatArray, results["mean_test_score"])[indices]
    maximum = float(np.max(scores))
    tied_local = np.flatnonzero(
        np.isclose(scores, maximum, rtol=_SELECTION_RTOL, atol=_SELECTION_ATOL)
    )
    tied = indices[tied_local]
    ranks = cast(IntArray, results["predictor_rank"])[tied]
    return int(tied[int(np.argmin(ranks))])


def _path_surface(
    h_values: IntArray,
    r_values: IntArray,
    evaluated_h: IntArray,
    evaluated_r: IntArray,
    values: FloatArray,
) -> FloatArray:
    surface = np.full((h_values.size, r_values.size), np.nan, dtype=np.float64)
    h_index = {int(value): index for index, value in enumerate(h_values)}
    r_index = {int(value): index for index, value in enumerate(r_values)}
    for h, r, value in zip(evaluated_h, evaluated_r, values, strict=True):
        surface[h_index[int(h)], r_index[int(r)]] = value
    return surface


def _validated_integer_values(
    values: Sequence[int] | None,
    *,
    name: str,
    lower: int,
    upper: int,
) -> IntArray:
    if values is None:
        return np.arange(lower, upper + 1, dtype=np.intp)
    array = np.asarray(values)
    if array.ndim != 1 or array.size == 0 or array.dtype.kind not in "iu":
        raise ValueError(f"{name} must be a nonempty one-dimensional integer sequence.")
    converted = np.asarray(array, dtype=np.intp)
    if np.unique(converted).size != converted.size:
        raise ValueError(f"{name} must not contain duplicate values.")
    if np.any(converted < lower) or np.any(converted > upper):
        raise ValueError(f"{name} values must lie in [{lower}, {upper}].")
    return np.sort(converted)


def _validate_optional_integer_sequence(values: object, *, name: str) -> None:
    if values is None:
        return
    if isinstance(values, (str, bytes)):
        raise ValueError(f"{name} must be None or a sequence of positive integers.")
    try:
        sequence = tuple(cast(Iterable[object], values))
    except TypeError as error:
        raise ValueError(f"{name} must be None or a sequence of positive integers.") from error
    if not sequence:
        raise ValueError(f"{name} must not be empty.")
    for value in sequence:
        _validate_positive_int(value, name=name)


def _resolve_path_scorer(scoring: str | Scorer) -> Scorer | None:
    if isinstance(scoring, str):
        if scoring == _DEFAULT_SCORING:
            return None
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


def _validate_cv(cv: object) -> None:
    if isinstance(cv, (bool, np.bool_)):
        raise ValueError("cv must be an integer at least 2, a splitter, or an iterable of splits.")
    if isinstance(cv, (int, np.integer)) and int(cv) < 2:
        raise ValueError("cv must be at least 2 when supplied as an integer.")
    if cv is None or isinstance(cv, (float, np.floating, str, bytes)):
        raise ValueError("cv must be an integer at least 2, a splitter, or an iterable of splits.")
