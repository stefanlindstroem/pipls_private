"""Public Pi-PLS estimator with fixed, exhaustive, and adaptive rank modes."""

from __future__ import annotations

import warnings
from collections.abc import Callable
from typing import Any, Literal, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.base import (
    BaseEstimator,
    ClassNamePrefixFeaturesOutMixin,
    MultiOutputMixin,
    RegressorMixin,
    TransformerMixin,
)
from sklearn.metrics import check_scoring, get_scorer, r2_score
from sklearn.utils.validation import check_array, check_is_fitted

from ._core import ResolvedSVDSolver, SVDSolver, fit_pipls_core
from ._cv_engine import (
    _evaluate_candidate_batch,
    _PiPLSCandidate,
    _PiPLSCandidateResult,
)
from ._sklearn_compat import _validate_estimator_data
from .decomposition import PiPLSDecomposition
from .exceptions import StatisticalSupportWarning
from .metrics import neg_response_standardized_mean_squared_error
from .model_selection import (
    _as_positive_float,
    _materialize_cv_splits,
    _max_predictor_rank,
    _predictor_rank_values,
    _rank_test_scores,
    _search_predictor_ranks,
    _select_predictor_rank,
    _training_response_scale,
)

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.intp]
Scorer = Callable[[Any, ArrayLike, ArrayLike], float]
Scoring = str | Scorer | None
_DEFAULT_SCORING = "neg_response_standardized_mean_squared_error"
_MIN_TRUSTED_SAMPLES_PER_PREDICTOR_RANK = 5.0
_MAX_RANDOM_STATE = int(np.iinfo(np.uint32).max)


class PiPLSRegression(
    ClassNamePrefixFeaturesOutMixin,  # type: ignore[misc]
    TransformerMixin,  # type: ignore[misc]
    RegressorMixin,  # type: ignore[misc]
    MultiOutputMixin,  # type: ignore[misc]
    BaseEstimator,  # type: ignore[misc]
):
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
        Rule-based values below 5 emit ``StatisticalSupportWarning``. It does
        not constrain an explicitly supplied integer rank.
    cv:
        Cross-validation splitter, integer split count of at least 2, iterable
        of train-validation index pairs, or ``None`` for the standard five-fold
        regression split, used when ``predictor_rank`` is ``"auto"`` or ``"optimal"``.
    scoring:
        Scikit-learn scorer name, callable, or ``None`` to use estimator ``score``.
        The default is negative response-standardized mean squared error.
    n_jobs:
        Nonzero integer number of predictor-rank candidates evaluated concurrently
        in cross-validated modes. ``None`` uses joblib's default and ``-1`` uses
        all processors.
    svd_solver:
        Predictor SVD policy. ``"full"`` uses the exact thin SVD,
        ``"randomized"`` always uses randomized truncated SVD, and ``"auto"``
        uses randomized SVD only for sufficiently large matrices and low retained
        rank. The response-side and coupling SVDs always remain exact.
    random_state:
        Integer seed in $[0, 2^{32}-1]$ used by randomized SVD. The default makes
        ``"auto"`` and ``"randomized"`` reproducible. ``None`` is accepted only
        with ``svd_solver="full"``.
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
        scoring: Scoring = _DEFAULT_SCORING,
        n_jobs: int | None = None,
        svd_solver: SVDSolver = "auto",
        random_state: int | None = 0,
    ) -> None:
        self.n_components = n_components
        self.scale = scale
        self.copy = copy
        self.predictor_rank = predictor_rank
        self.samples_per_predictor_rank = samples_per_predictor_rank
        self.cv = cv
        self.scoring = scoring
        self.n_jobs = n_jobs
        self.svd_solver = svd_solver
        self.random_state = random_state

    def fit(self, X: ArrayLike, y: ArrayLike) -> PiPLSRegression:
        """Fit the Pi-PLS model and select predictor rank when requested."""

        self._validate_constructor_parameters()
        validated = _validate_estimator_data(
            self,
            X,
            y,
            reset=True,
            accept_sparse=False,
            dtype=np.float64,
            multi_output=True,
            y_numeric=True,
            ensure_min_samples=2,
            copy=self.copy,
        )
        X_checked, y_checked = cast(tuple[Any, Any], validated)
        X_array = np.asarray(X_checked, dtype=np.float64)
        y_array_raw = np.array(y_checked, dtype=np.float64, copy=self.copy)
        self._y_was_1d = y_array_raw.ndim == 1
        y_array = y_array_raw.reshape(-1, 1) if self._y_was_1d else y_array_raw

        self.n_targets_ = int(y_array.shape[1])
        if self.n_components > self.n_targets_:
            raise ValueError(
                "n_components must not exceed the number of response columns: "
                f"got n_components={self.n_components}, n_targets={self.n_targets_}."
            )
        self._clear_selection_attributes()

        if isinstance(self.predictor_rank, str) and self.predictor_rank in ("auto", "optimal"):
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

    def predict(self, X: ArrayLike, copy: bool = True) -> FloatArray:
        """Predict responses in their original units.

        Parameters
        ----------
        X:
            Predictor matrix.
        copy:
            Whether validation may copy ``X``. This mirrors
            :class:`sklearn.cross_decomposition.PLSRegression`.
        """

        check_is_fitted(self, attributes=["coef_", "intercept_"])
        X_checked = cast(
            FloatArray,
            _validate_estimator_data(
                self,
                X,
                reset=False,
                accept_sparse=False,
                dtype=np.float64,
                copy=copy,
            ),
        )
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
        copy: bool = True,
    ) -> FloatArray | tuple[FloatArray, FloatArray]:
        """Transform predictors, and optionally responses, to latent scores."""

        check_is_fitted(self, attributes=["P_", "Q_", "x_mean_", "y_mean_"])
        X_checked = cast(
            FloatArray,
            _validate_estimator_data(
                self,
                X,
                reset=False,
                accept_sparse=False,
                dtype=np.float64,
                copy=copy,
            ),
        )
        X_cs = (np.asarray(X_checked, dtype=np.float64) - self.x_mean_) / self.x_scale_
        x_scores = cast(FloatArray, X_cs @ self.P_)
        if y is None:
            return x_scores

        y_checked = check_array(
            y,
            ensure_2d=False,
            dtype=np.float64,
            ensure_min_samples=X_cs.shape[0],
            copy=copy,
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

    def fit_transform(
        self,
        X: ArrayLike,
        y: ArrayLike | None = None,
    ) -> tuple[FloatArray, FloatArray]:
        """Fit the model and return predictor and response scores.

        As for scikit-learn's ``PLSRegression``, supplying ``y`` returns the
        pair ``(x_scores, y_scores)``.
        """

        if y is None:
            raise ValueError("y is required to fit PiPLSRegression.")
        self.fit(X, y)
        return self.x_scores_.copy(), self.y_scores_.copy()

    def inverse_transform(
        self,
        X: ArrayLike,
        y: ArrayLike | None = None,
    ) -> FloatArray | tuple[FloatArray, FloatArray]:
        """Reconstruct predictors, and optionally responses, from latent scores.

        Reconstruction is least-squares and is exact only when the retained
        latent spaces span the corresponding centered/scaled data spaces.
        """

        check_is_fitted(self, attributes=["x_loadings_", "y_loadings_"])
        x_scores = check_array(X, ensure_2d=True, dtype=np.float64)
        if x_scores.shape[1] != self.n_components:
            raise ValueError(
                "X has an incompatible number of latent components: "
                f"expected {self.n_components}, got {x_scores.shape[1]}."
            )
        X_original = (x_scores @ self.x_loadings_.T) * self.x_scale_ + self.x_mean_
        if y is None:
            return np.asarray(X_original, dtype=np.float64)

        y_scores = check_array(y, ensure_2d=False, dtype=np.float64)
        if y_scores.ndim == 1:
            y_scores = y_scores.reshape(-1, 1)
        if y_scores.shape[0] != x_scores.shape[0]:
            raise ValueError(
                "X and y scores must contain the same number of samples: "
                f"got {x_scores.shape[0]} and {y_scores.shape[0]}."
            )
        if y_scores.shape[1] != self.n_components:
            raise ValueError(
                "y has an incompatible number of latent components: "
                f"expected {self.n_components}, got {y_scores.shape[1]}."
            )
        y_original = (y_scores @ self.y_loadings_.T) * self.y_scale_ + self.y_mean_
        return (
            np.asarray(X_original, dtype=np.float64),
            np.asarray(y_original, dtype=np.float64),
        )

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

    def _more_tags(self) -> dict[str, bool]:
        """Legacy scikit-learn tags for releases before the Tags dataclasses."""

        return {"multioutput": True, "poor_score": True}

    def __sklearn_tags__(self) -> Any:
        """Declare multi-output regression and PLS-like score expectations."""

        parent = getattr(super(), "__sklearn_tags__", None)
        if parent is None:  # pragma: no cover - scikit-learn 1.4/1.5
            return self._more_tags()
        tags = parent()
        tags.target_tags.multi_output = True
        tags.target_tags.single_output = True
        if tags.regressor_tags is not None:
            tags.regressor_tags.poor_score = True
        return tags

    def _select_cross_validated_rank(
        self,
        X: FloatArray,
        y: FloatArray,
        *,
        search_method: Literal["auto", "optimal"],
    ) -> tuple[int, int]:
        materialized = _materialize_cv_splits(_validated_cv(self.cv), X, y)
        max_predictor_rank = _max_predictor_rank(
            n_features=self.n_features_in_,
            n_train_min=materialized.n_train_min,
            samples_per_predictor_rank=self.samples_per_predictor_rank,
        )
        admissible_ranks = _predictor_rank_values(
            n_components=self.n_components,
            max_predictor_rank=max_predictor_rank,
        )
        scorer = _resolve_scorer(self.scoring, self)
        self.scorer_ = (
            neg_response_standardized_mean_squared_error
            if _uses_default_scoring(self.scoring)
            else scorer
        )
        cache: dict[tuple[int, int], _PiPLSCandidateResult] = {}
        evaluation_order: list[int] = []

        def evaluate(ranks: IntArray) -> IntArray:
            candidates = tuple(
                _PiPLSCandidate(
                    n_components=self.n_components,
                    predictor_rank=int(rank),
                )
                for rank in ranks
            )
            evaluated = _evaluate_candidate_batch(
                candidates=candidates,
                cache=cache,
                template=self,
                n_components_key="n_components",
                predictor_rank_key="predictor_rank",
                scorer=scorer,
                use_default_scoring=_uses_default_scoring(self.scoring),
                X=X,
                y=y,
                splits=materialized.splits,
                n_jobs=self.n_jobs,
                solver_getter=_regression_solver,
                parallel_preference="threads",
            )
            evaluated_ranks = np.asarray(
                [candidate.predictor_rank for candidate in evaluated],
                dtype=np.intp,
            )
            evaluation_order.extend(int(rank) for rank in evaluated_ranks)
            return evaluated_ranks

        def evaluated_scores() -> tuple[IntArray, FloatArray]:
            ranks = np.asarray(
                sorted(
                    predictor_rank
                    for n_components, predictor_rank in cache
                    if n_components == self.n_components
                ),
                dtype=np.intp,
            )
            scores = np.asarray(
                [
                    np.mean(cache[(self.n_components, int(rank))].split_scores)
                    for rank in ranks
                ],
                dtype=np.float64,
            )
            return ranks, scores

        search = _search_predictor_ranks(
            allowed_ranks=admissible_ranks,
            search_method=search_method,
            evaluate=evaluate,
            evaluated_scores=evaluated_scores,
        )

        predictor_rank_values, mean_scores = evaluated_scores()
        evaluations = [
            cache[(self.n_components, int(rank))]
            for rank in predictor_rank_values
        ]
        split_scores = np.vstack([result.split_scores for result in evaluations])
        split_mse = np.vstack(
            [result.split_response_standardized_mse for result in evaluations]
        )
        split_fit_times = np.vstack([result.split_fit_times for result in evaluations])
        split_score_times = np.vstack([result.split_score_times for result in evaluations])
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
        self.predictor_rank_search_history_ = tuple(
            batch.copy() for batch in search.history
        )
        self.predictor_rank_cv_svd_solvers_ = {
            result.predictor_rank: result.split_svd_solvers for result in evaluations
        }
        self.cv_results_ = _cv_results_dictionary(
            n_components=self.n_components,
            predictor_rank_values=predictor_rank_values,
            split_scores=split_scores,
            split_response_standardized_mse=split_mse,
            split_fit_times=split_fit_times,
            split_score_times=split_score_times,
        )
        self.predictor_rank_cv_results_ = self.cv_results_
        self.predictor_rank_search_method_ = search_method
        self.predictor_rank_search_interval_ = np.asarray(
            search.final_interval,
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
        self.best_index_ = selected_index
        self.best_params_ = {
            "n_components": self.n_components,
            "predictor_rank": selected_rank,
        }
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
        self.response_scale_for_scoring_ = _training_response_scale(y)
        X -= self.x_mean_
        y -= self.y_mean_

        if self.scale:
            self.x_scale_ = _safe_sample_scale(X)
            self.y_scale_ = _safe_sample_scale(y)
            X /= self.x_scale_
            y /= self.y_scale_
        else:
            self.x_scale_ = np.ones(X.shape[1], dtype=np.float64)
            self.y_scale_ = np.ones(y.shape[1], dtype=np.float64)

        X_cs = X
        y_cs = y
        result = fit_pipls_core(
            X_cs,
            y_cs,
            predictor_rank=predictor_rank,
            n_components=self.n_components,
            svd_solver=self.svd_solver,
            random_state=self.random_state,
        )

        self.predictor_rank_ = predictor_rank
        self.max_predictor_rank_ = max_predictor_rank
        self.decomposition_ = PiPLSDecomposition._from_core_result(result)
        self.Pi_ = self.decomposition_.Pi
        self.C_ = self.decomposition_.C
        self.W_ = self.decomposition_.W
        self.P_ = self.decomposition_.P
        self.D_ = self.decomposition_.D
        self.dilation_ = self.decomposition_.dilation
        self.Q_ = self.decomposition_.Q
        self.x_rotations_ = self.P_
        self.y_rotations_ = self.Q_
        self.x_weights_ = self.P_
        self.y_weights_ = self.Q_
        self._n_features_out = self.n_components
        self.x_rank_ = result.x_rank
        self.x_rank_is_exact_ = result.x_rank_is_exact
        self.rank_tolerance_ = result.rank_tolerance
        self.svd_solver_ = result.predictor_svd_solver

        self.coef_matrix_ = result.regression_map * self.y_scale_[None, :] / self.x_scale_[:, None]
        self.coef_ = self.coef_matrix_.T
        self.intercept_ = self.y_mean_ - self.x_mean_ @ self.coef_matrix_
        self.x_scores_ = X_cs @ self.P_
        self.y_scores_ = y_cs @ self.Q_
        x_loadings, _, _, _ = np.linalg.lstsq(self.x_scores_, X_cs, rcond=None)
        y_loadings, _, _, _ = np.linalg.lstsq(self.y_scores_, y_cs, rcond=None)
        self.x_loadings_ = np.asarray(x_loadings.T, dtype=np.float64)
        self.y_loadings_ = np.asarray(y_loadings.T, dtype=np.float64)

    def _validate_constructor_parameters(self) -> None:
        _validate_positive_int(self.n_components, name="n_components")
        uses_rank_rule = isinstance(self.predictor_rank, str) and self.predictor_rank in (
            "max",
            "optimal",
            "auto",
        )
        if uses_rank_rule:
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
        samples_per_rank = _as_positive_float(
            self.samples_per_predictor_rank,
            name="samples_per_predictor_rank",
        )
        if not isinstance(self.scale, (bool, np.bool_)):
            raise ValueError(f"scale must be boolean; got {self.scale!r}.")
        if not isinstance(self.copy, (bool, np.bool_)):
            raise ValueError(f"copy must be boolean; got {self.copy!r}.")
        if not isinstance(self.svd_solver, str) or self.svd_solver not in (
            "full",
            "randomized",
            "auto",
        ):
            raise ValueError(
                'svd_solver must be "full", "randomized", or "auto"; '
                f"got {self.svd_solver!r}."
            )
        _validated_cv(self.cv)
        _validate_n_jobs(self.n_jobs)
        _validate_random_state(self.random_state, svd_solver=self.svd_solver)
        if self.predictor_rank in ("auto", "optimal"):
            _resolve_scorer(self.scoring, self)
        if uses_rank_rule and samples_per_rank < _MIN_TRUSTED_SAMPLES_PER_PREDICTOR_RANK:
            warnings.warn(
                f"samples_per_predictor_rank={samples_per_rank:g} is below 5. "
                "This permits fewer than five training samples per retained predictor-rank "
                "direction, so the resulting rank bound may not have sufficient statistical "
                "support to be trusted without external validation.",
                StatisticalSupportWarning,
                stacklevel=3,
            )

    def _clear_selection_attributes(self) -> None:
        for name in (
            "predictor_rank_values_",
            "scorer_",
            "predictor_rank_cv_results_",
            "cv_results_",
            "best_index_",
            "best_params_",
            "predictor_rank_cv_svd_solvers_",
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


def _regression_solver(estimator: Any) -> ResolvedSVDSolver:
    """Return the resolved predictor solver from a fitted Pi-PLS estimator."""

    return cast(ResolvedSVDSolver, estimator.svd_solver_)


def _uses_default_scoring(scoring: Scoring) -> bool:
    """Return whether the shared default response-standardized scorer is requested."""

    return isinstance(scoring, str) and scoring == _DEFAULT_SCORING


def _cv_results_dictionary(
    *,
    n_components: int,
    predictor_rank_values: NDArray[np.intp],
    split_scores: FloatArray,
    split_response_standardized_mse: FloatArray,
    split_fit_times: FloatArray,
    split_score_times: FloatArray,
) -> dict[str, Any]:
    n_candidates = predictor_rank_values.size
    n_components_values = np.full(n_candidates, n_components, dtype=np.intp)
    mean_scores = np.mean(split_scores, axis=1)
    results: dict[str, Any] = {
        "params": [
            {"n_components": n_components, "predictor_rank": int(rank)}
            for rank in predictor_rank_values
        ],
        "n_components": n_components_values,
        "predictor_rank": predictor_rank_values.copy(),
        "param_n_components": n_components_values.copy(),
        "param_predictor_rank": predictor_rank_values.copy(),
        "mean_test_score": mean_scores,
        "std_test_score": np.std(split_scores, axis=1),
        "mean_fit_time": np.mean(split_fit_times, axis=1),
        "std_fit_time": np.std(split_fit_times, axis=1),
        "mean_score_time": np.mean(split_score_times, axis=1),
        "std_score_time": np.std(split_score_times, axis=1),
        "mean_response_standardized_mse": np.mean(
            split_response_standardized_mse,
            axis=1,
        ),
        "std_response_standardized_mse": np.std(
            split_response_standardized_mse,
            axis=1,
        ),
    }
    results["rank_test_score"] = _rank_test_scores(mean_scores)
    for split_index in range(split_scores.shape[1]):
        results[f"split{split_index}_test_score"] = split_scores[:, split_index].copy()
        results[f"split{split_index}_response_standardized_mse"] = (
            split_response_standardized_mse[:, split_index].copy()
        )
    return results


def _resolve_scorer(scoring: Scoring, estimator: Any) -> Scorer | None:
    if isinstance(scoring, str):
        if scoring == _DEFAULT_SCORING:
            return None
        try:
            return cast(Scorer, get_scorer(scoring))
        except ValueError as error:
            raise ValueError(f"Unknown scoring value {scoring!r}.") from error
    if scoring is None:
        return cast(Scorer, check_scoring(estimator, scoring=None))
    if callable(scoring):
        return scoring
    raise ValueError(
        "scoring must be None, a scikit-learn scorer name, or callable; "
        f"got {scoring!r}."
    )


def _validated_cv(cv: object) -> object:
    if isinstance(cv, (bool, np.bool_)):
        raise ValueError(
            "cv must be an integer at least 2, a cross-validation splitter, "
            f"or an iterable of splits; got {cv!r}."
        )
    if isinstance(cv, (int, np.integer)):
        n_splits = int(cv)
        if n_splits < 2:
            raise ValueError(f"cv must be at least 2 when supplied as an integer; got {cv!r}.")
        return n_splits
    if cv is None:
        return None
    if isinstance(
        cv,
        (float, np.floating, complex, np.complexfloating, str, bytes),
    ):
        raise ValueError(
            "cv must be an integer at least 2, a cross-validation splitter, "
            f"or an iterable of splits; got {cv!r}."
        )
    return cv


def _validate_n_jobs(n_jobs: int | None) -> None:
    if n_jobs is None:
        return
    if isinstance(n_jobs, (bool, np.bool_)) or not isinstance(n_jobs, (int, np.integer)):
        raise ValueError(f"n_jobs must be None or a nonzero integer; got {n_jobs!r}.")
    if int(n_jobs) == 0:
        raise ValueError("n_jobs must not be zero.")



def _validate_random_state(random_state: int | None, *, svd_solver: SVDSolver) -> None:
    if random_state is None:
        if svd_solver in ("randomized", "auto"):
            raise ValueError(
                "random_state must be an integer between 0 and "
                f"{_MAX_RANDOM_STATE} when svd_solver={svd_solver!r}."
            )
        return
    if isinstance(random_state, (bool, np.bool_)) or not isinstance(
        random_state,
        (int, np.integer),
    ):
        raise ValueError(
            "random_state must be None or an integer between 0 and "
            f"{_MAX_RANDOM_STATE}; got {random_state!r}."
        )
    seed = int(random_state)
    if seed < 0 or seed > _MAX_RANDOM_STATE:
        raise ValueError(
            "random_state must be None or an integer between 0 and "
            f"{_MAX_RANDOM_STATE}; got {random_state!r}."
        )


def _safe_sample_scale(centered: FloatArray) -> FloatArray:
    if centered.shape[0] <= 1:
        return np.ones(centered.shape[1], dtype=np.float64)
    scale = np.std(centered, axis=0, ddof=1)
    return np.where(scale == 0.0, 1.0, scale).astype(np.float64, copy=False)


def _validate_positive_int(value: Any, *, name: str) -> None:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be a positive integer; got {value!r}.")
    if int(value) < 1:
        raise ValueError(f"{name} must be a positive integer; got {value!r}.")
