"""Private shared cross-validation engine for Pi-PLS candidate searches."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any, Literal, cast

import numpy as np
from joblib import Parallel, delayed
from numpy.typing import ArrayLike, NDArray
from sklearn.base import clone
from sklearn.utils import _safe_indexing

from ._core import ResolvedSVDSolver
from .model_selection import CVSplit, _response_standardized_mse, _training_response_scale

FloatArray = NDArray[np.float64]
Scorer = Callable[[Any, ArrayLike, ArrayLike], float]
SolverGetter = Callable[[Any], ResolvedSVDSolver]
ParallelPreference = Literal["threads"] | None


@dataclass(frozen=True)
class _PiPLSCandidate:
    """One admissible Pi-PLS hyperparameter pair."""

    n_components: int
    predictor_rank: int

    @property
    def key(self) -> tuple[int, int]:
        """Return the stable cache key for this candidate."""

        return self.n_components, self.predictor_rank


@dataclass(frozen=True)
class _PiPLSCandidateResult:
    """Fold-level results for one Pi-PLS candidate."""

    candidate: _PiPLSCandidate
    split_scores: FloatArray
    split_response_standardized_mse: FloatArray
    split_svd_solvers: tuple[ResolvedSVDSolver, ...]

    @property
    def n_components(self) -> int:
        """Return the response-side component count."""

        return self.candidate.n_components

    @property
    def predictor_rank(self) -> int:
        """Return the predictor truncation rank."""

        return self.candidate.predictor_rank


CandidateCache = dict[tuple[int, int], _PiPLSCandidateResult]


def _evaluate_candidate_batch(
    *,
    candidates: Iterable[_PiPLSCandidate],
    cache: CandidateCache,
    template: Any,
    n_components_key: str,
    predictor_rank_key: str,
    scorer: Scorer | None,
    use_default_scoring: bool,
    X: ArrayLike,
    y: ArrayLike,
    splits: tuple[CVSplit, ...],
    n_jobs: int | None,
    solver_getter: SolverGetter | None = None,
    parallel_preference: ParallelPreference = None,
) -> tuple[_PiPLSCandidate, ...]:
    """Evaluate uncached candidates and update ``cache`` in input order."""

    pending: list[_PiPLSCandidate] = []
    seen: set[tuple[int, int]] = set()
    for candidate in candidates:
        if candidate.key in cache or candidate.key in seen:
            continue
        pending.append(candidate)
        seen.add(candidate.key)
    if not pending:
        return ()

    results = cast(
        list[_PiPLSCandidateResult],
        Parallel(n_jobs=n_jobs, prefer=parallel_preference)(
            delayed(_evaluate_candidate)(
                candidate=candidate,
                template=template,
                n_components_key=n_components_key,
                predictor_rank_key=predictor_rank_key,
                scorer=scorer,
                use_default_scoring=use_default_scoring,
                X=X,
                y=y,
                splits=splits,
                solver_getter=solver_getter,
            )
            for candidate in pending
        ),
    )
    for result in results:
        cache[result.candidate.key] = result
    return tuple(pending)


def _evaluate_candidate(
    *,
    candidate: _PiPLSCandidate,
    template: Any,
    n_components_key: str,
    predictor_rank_key: str,
    scorer: Scorer | None,
    use_default_scoring: bool,
    X: ArrayLike,
    y: ArrayLike,
    splits: tuple[CVSplit, ...],
    solver_getter: SolverGetter | None,
) -> _PiPLSCandidateResult:
    """Evaluate one candidate on every materialized split."""

    split_scores = np.empty(len(splits), dtype=np.float64)
    split_mse = np.empty(len(splits), dtype=np.float64)
    split_svd_solvers: list[ResolvedSVDSolver] = []
    params = {
        n_components_key: candidate.n_components,
        predictor_rank_key: candidate.predictor_rank,
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

        if use_default_scoring:
            split_scores[split_index] = -mse
        else:
            assert scorer is not None
            score = float(scorer(estimator, X_validation, y_validation))
            if not np.isfinite(score):
                raise ValueError(
                    "The scoring callable returned a nonfinite value for "
                    f"n_components={candidate.n_components}, "
                    f"predictor_rank={candidate.predictor_rank}, split={split_index}."
                )
            split_scores[split_index] = score

        if solver_getter is not None:
            split_svd_solvers.append(solver_getter(estimator))

    return _PiPLSCandidateResult(
        candidate=candidate,
        split_scores=split_scores,
        split_response_standardized_mse=split_mse,
        split_svd_solvers=tuple(split_svd_solvers),
    )
