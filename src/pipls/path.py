"""Pipeline-aware cross-validated Pi-PLS path analysis."""

from __future__ import annotations

import warnings
from collections.abc import Callable, Iterable, Sequence
from time import perf_counter
from typing import Any, Literal, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.base import (
    BaseEstimator,
    MetaEstimatorMixin,
    MultiOutputMixin,
    RegressorMixin,
    TransformerMixin,
    clone,
)
from sklearn.metrics import check_scoring, get_scorer
from sklearn.pipeline import Pipeline
from sklearn.utils import _safe_indexing, indexable
from sklearn.utils.metaestimators import available_if
from sklearn.utils.validation import check_is_fitted

from ._cv_engine import (
    _evaluate_candidate_batch,
    _ordered_oof_predictions,
    _PiPLSCandidate,
    _PiPLSCandidateResult,
)
from ._sklearn_compat import _validate_estimator_data
from .exceptions import StatisticalSupportWarning
from .metrics import neg_response_standardized_mean_squared_error
from .model_selection import (
    _as_positive_float,
    _is_leave_one_out_splits,
    _materialize_cv_splits,
    _max_predictor_rank,
    _pooled_oof_r2,
    _rank_test_scores,
    _search_predictor_ranks,
    _validate_positive_int,
    _validate_singleton_fold_scoring,
)
from .regression import PiPLSRegression
from .validation import PiPLSValidationReport

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.intp]
Scorer = Callable[[Any, ArrayLike, ArrayLike], float]
Scoring = str | Scorer | None
SearchMethod = Literal["optimal", "auto"]
_DEFAULT_SCORING = "neg_response_standardized_mean_squared_error"
_MIN_TRUSTED_SAMPLES_PER_PREDICTOR_RANK = 5.0
_SELECTION_RTOL = 1e-12
_SELECTION_ATOL = 1e-15


def _estimator_supports(method_name: str) -> Callable[[Any], bool]:
    def check(search: Any) -> bool:
        if hasattr(search, "best_params_") and not hasattr(search, "best_estimator_"):
            return False
        estimator = (
            search.best_estimator_
            if hasattr(search, "best_estimator_")
            else (PiPLSRegression() if search.estimator is None else search.estimator)
        )
        return hasattr(estimator, method_name)

    return check


class PiPLSPathCV(
    TransformerMixin,  # type: ignore[misc]
    RegressorMixin,  # type: ignore[misc]
    MultiOutputMixin,  # type: ignore[misc]
    MetaEstimatorMixin,  # type: ignore[misc]
    BaseEstimator,  # type: ignore[misc]
):
    r"""Cross-validated search over the admissible Pi-PLS rank path.

    The default ``search_method="auto"`` applies the same deterministic logarithmic
    coarse-to-fine predictor-rank search used by :class:`PiPLSRegression`
    independently for each value of ``n_components``.

    Parameters
    ----------
    estimator:
        A direct :class:`PiPLSRegression` or a scikit-learn :class:`Pipeline`
        whose final step is :class:`PiPLSRegression`. ``None`` creates a default
        direct estimator template.
    pipls_param_prefix:
        Optional final pipeline-step name locating Pi-PLS. It is inferred for
        supported pipelines and retained for explicitness and nested parameter
        compatibility.
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
        Integer split count, splitter, iterable of train-validation pairs, or
        ``None`` for the standard five-fold regression split.
    scoring:
        Scikit-learn scorer name, callable, or ``None`` to use estimator ``score``.
        The default is negative response-standardized MSE.
    refit:
        Refit the globally selected pair on all supplied data.
    n_jobs:
        Joblib parallelism across candidate pairs within each evaluation batch.
    return_oof_predictions:
        If true, fit the selected fixed parameterization on every training fold
        and retain row-ordered validation predictions. Repeated predictions are
        averaged and rows never validated are marked with NaN.
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
        scoring: Scoring = _DEFAULT_SCORING,
        refit: bool = True,
        n_jobs: int | None = None,
        return_oof_predictions: bool = False,
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
        self.return_oof_predictions = return_oof_predictions

    def fit(
        self,
        X: ArrayLike,
        y: ArrayLike,
        *,
        groups: ArrayLike | None = None,
    ) -> PiPLSPathCV:
        """Evaluate the path and optionally refit the globally selected pair."""

        self._validate_constructor_parameters()
        self._clear_validation_attributes()
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
            copy=True,
        )
        X_checked, y_checked = cast(tuple[Any, Any], validated)
        X_array = np.asarray(X_checked, dtype=np.float64)
        y_array = np.array(y_checked, dtype=np.float64, copy=True)
        y_2d = y_array.reshape(-1, 1) if y_array.ndim == 1 else y_array
        self.n_targets_ = int(y_2d.shape[1])
        X_indexable, y_indexable = indexable(X, y)

        template = PiPLSRegression() if self.estimator is None else self.estimator
        self.pipls_param_prefix_ = _resolve_pipls_param_prefix(
            template,
            self.pipls_param_prefix,
        )
        n_components_key, predictor_rank_key = _pipls_parameter_keys(
            self.pipls_param_prefix_
        )
        return_oof_key = _pipls_parameter_key(
            self.pipls_param_prefix_,
            "return_oof_predictions",
        )
        template = clone(template).set_params(**{return_oof_key: False})
        materialized = _materialize_cv_splits(self.cv, X_array, y_array, groups=groups)
        _validate_singleton_fold_scoring(self.scoring, materialized.splits)
        self.n_splits_ = len(materialized.splits)
        self.cv_n_train_min_ = materialized.n_train_min

        fold_feature_limit = _fold_safe_feature_limit(
            template=template,
            prefix=self.pipls_param_prefix_,
            X=X_indexable,
            y=y_indexable,
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

        scorer = _resolve_path_scorer(self.scoring, template)
        self.scorer_ = (
            neg_response_standardized_mean_squared_error
            if _uses_default_path_scoring(self.scoring)
            else scorer
        )
        cache: dict[tuple[int, int], _PiPLSCandidateResult] = {}
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
                X=X_indexable,
                y=y_indexable,
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
                    X=X_indexable,
                    y=y_indexable,
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

        self.best_pipls_params_ = {
            "n_components": self.best_n_components_,
            "predictor_rank": self.best_predictor_rank_,
        }
        predictions: FloatArray | None = None
        counts: IntArray | None = None
        pooled_r2: float | None = None
        if self.return_oof_predictions:
            oof = _ordered_oof_predictions(
                candidate=_PiPLSCandidate(
                    n_components=self.best_n_components_,
                    predictor_rank=self.best_predictor_rank_,
                ),
                template=template,
                n_components_key=n_components_key,
                predictor_rank_key=predictor_rank_key,
                scorer=scorer,
                use_default_scoring=_uses_default_path_scoring(self.scoring),
                X=X_indexable,
                y=y_indexable,
                splits=materialized.splits,
                n_jobs=self.n_jobs,
            )
            predictions = oof.predictions[:, 0] if y_array.ndim == 1 else oof.predictions
            counts = oof.prediction_counts
            pooled_r2 = _pooled_oof_r2(y_indexable, predictions, counts)

        self.validation_report_ = PiPLSValidationReport(
            n_components=self.best_n_components_,
            predictor_rank=self.best_predictor_rank_,
            n_splits=self.n_splits_,
            mean_test_score=self.best_score_,
            mean_response_standardized_mse=self.best_response_standardized_mse_,
            estimate_kind="selection-conditioned",
            is_leave_one_out=_is_leave_one_out_splits(
                materialized.splits,
                n_samples=int(y_array.shape[0]),
            ),
            oof_predictions=predictions,
            oof_prediction_counts=counts,
            pooled_oof_r2=pooled_r2,
        )
        if self.return_oof_predictions:
            assert self.validation_report_.oof_predictions is not None
            assert self.validation_report_.oof_prediction_counts is not None
            self.oof_predictions_ = self.validation_report_.oof_predictions
            self.oof_prediction_counts_ = self.validation_report_.oof_prediction_counts
            self.pooled_oof_r2_ = self.validation_report_.pooled_oof_r2
            self.oof_params_ = self.best_params_.copy()

        if self.refit:
            refit_started = perf_counter()
            self.best_estimator_ = clone(template).set_params(**self.best_params_)
            self.best_estimator_.fit(X_indexable, y_indexable)
            self.refit_time_ = perf_counter() - refit_started
            self.best_pipls_ = _extract_fitted_pipls(
                self.best_estimator_,
                self.pipls_param_prefix_,
            )
        else:
            for name in ("best_estimator_", "best_pipls_", "refit_time_"):
                if hasattr(self, name):
                    delattr(self, name)
        return self

    def predict(self, X: ArrayLike, copy: bool = True) -> FloatArray:
        """Predict with the refitted globally selected estimator."""

        estimator = self._refitted_estimator()
        if isinstance(estimator, PiPLSRegression):
            return estimator.predict(X, copy=copy)
        return cast(FloatArray, estimator.predict(X))

    @available_if(_estimator_supports("transform"))  # type: ignore[untyped-decorator]
    def transform(
        self,
        X: ArrayLike,
        y: ArrayLike | None = None,
        copy: bool = True,
    ) -> Any:
        """Transform with the refitted globally selected estimator."""

        estimator = self._refitted_estimator()
        if isinstance(estimator, PiPLSRegression):
            return estimator.transform(X, y, copy=copy)
        if y is not None:
            raise ValueError(
                "transform(X, y) is available when the refitted estimator is a direct "
                "PiPLSRegression. For composite estimators, use best_pipls_ with data "
                "transformed by the preceding pipeline steps."
            )
        return estimator.transform(X)

    @available_if(_estimator_supports("transform"))  # type: ignore[untyped-decorator]
    def fit_transform(
        self,
        X: ArrayLike,
        y: ArrayLike | None = None,
        *,
        groups: ArrayLike | None = None,
        **fit_params: Any,
    ) -> Any:
        """Fit the path search and delegate transformation to the selected estimator."""

        if fit_params:
            names = ", ".join(sorted(fit_params))
            raise TypeError(f"Unexpected fit parameters: {names}.")
        if y is None:
            raise ValueError("y is required to fit PiPLSPathCV.")
        self.fit(X, y, groups=groups)
        if isinstance(self.best_estimator_, PiPLSRegression):
            return cast(tuple[FloatArray, FloatArray], self.transform(X, y))
        return cast(Any, self.transform(X))

    @available_if(_estimator_supports("inverse_transform"))  # type: ignore[untyped-decorator]
    def inverse_transform(
        self,
        X: ArrayLike,
        y: ArrayLike | None = None,
    ) -> Any:
        """Delegate inverse transformation to the refitted selected estimator."""

        estimator = self._refitted_estimator()
        if isinstance(estimator, PiPLSRegression):
            return estimator.inverse_transform(X, y)
        if y is not None:
            raise ValueError(
                "inverse_transform(X, y) is available when the refitted estimator "
                "is a direct PiPLSRegression. Composite estimators reconstruct X only."
            )
        return estimator.inverse_transform(X)

    def score(
        self,
        X: ArrayLike,
        y: ArrayLike,
        sample_weight: ArrayLike | None = None,
    ) -> float:
        """Return the selected estimator's uniformly averaged :math:`R^2`."""

        estimator = self._refitted_estimator()
        if sample_weight is None:
            return float(estimator.score(X, y))
        return float(estimator.score(X, y, sample_weight=sample_weight))

    @available_if(_estimator_supports("get_feature_names_out"))  # type: ignore[untyped-decorator]
    def get_feature_names_out(
        self,
        input_features: ArrayLike | None = None,
    ) -> NDArray[np.object_]:
        """Return names for the selected latent predictor scores."""

        estimator = self._refitted_estimator()
        method = getattr(estimator, "get_feature_names_out", None)
        if method is not None:
            return cast(NDArray[np.object_], method(input_features))
        return cast(NDArray[np.object_], self.best_pipls_.get_feature_names_out(input_features))

    def _more_tags(self) -> dict[str, bool]:
        """Legacy scikit-learn tags for releases before the Tags dataclasses."""

        return {"multioutput": True, "poor_score": True}

    def __sklearn_tags__(self) -> Any:
        """Declare a multi-output regression meta-estimator and transformer."""

        parent = getattr(super(), "__sklearn_tags__", None)
        if parent is None:  # pragma: no cover - scikit-learn 1.4/1.5
            return self._more_tags()
        tags = parent()
        tags.target_tags.multi_output = True
        tags.target_tags.single_output = True
        if tags.regressor_tags is not None:
            tags.regressor_tags.poor_score = True
        return tags

    def _refitted_estimator(self) -> Any:
        check_is_fitted(self, attributes=["best_params_"])
        if not hasattr(self, "best_estimator_"):
            raise AttributeError(
                "PiPLSPathCV was fitted with refit=False; predict, transform, and score "
                "require refit=True."
            )
        return self.best_estimator_

    def _clear_validation_attributes(self) -> None:
        for name in (
            "validation_report_",
            "oof_predictions_",
            "oof_prediction_counts_",
            "pooled_oof_r2_",
            "oof_params_",
        ):
            if hasattr(self, name):
                delattr(self, name)

    def _validate_constructor_parameters(self) -> None:
        if self.estimator is not None:
            _validate_supported_estimator(self.estimator)
        if self.pipls_param_prefix is not None and not isinstance(
            self.pipls_param_prefix, str
        ):
            raise ValueError("pipls_param_prefix must be None or a string.")
        if self.search_method not in ("optimal", "auto"):
            raise ValueError('search_method must be "optimal" or "auto".')
        if not isinstance(self.refit, (bool, np.bool_)):
            raise ValueError(f"refit must be boolean; got {self.refit!r}.")
        if not isinstance(self.return_oof_predictions, (bool, np.bool_)):
            raise ValueError(
                "return_oof_predictions must be boolean; "
                f"got {self.return_oof_predictions!r}."
            )
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
        scorer_template = (
            PiPLSRegression() if self.estimator is None else self.estimator
        )
        _resolve_path_scorer(self.scoring, scorer_template)


def _validate_supported_estimator(estimator: Any) -> None:
    if isinstance(estimator, PiPLSRegression):
        return
    if isinstance(estimator, Pipeline):
        if not estimator.steps or not isinstance(estimator.steps[-1][1], PiPLSRegression):
            raise ValueError(
                "estimator pipelines must end with a PiPLSRegression step."
            )
        return
    raise ValueError(
        "estimator must be PiPLSRegression or a sklearn Pipeline whose final "
        "step is PiPLSRegression."
    )


def _resolve_pipls_param_prefix(template: Any, supplied: str | None) -> str:
    _validate_supported_estimator(template)
    if isinstance(template, PiPLSRegression):
        if supplied not in (None, ""):
            raise ValueError(
                "pipls_param_prefix must be None or an empty string for a direct "
                "PiPLSRegression estimator."
            )
        return ""

    assert isinstance(template, Pipeline)
    final_name = template.steps[-1][0]
    if supplied is None:
        return cast(str, final_name)
    prefix = supplied.removesuffix("__")
    if prefix != final_name:
        raise ValueError(
            f"pipls_param_prefix={supplied!r} does not locate the final "
            f"PiPLSRegression pipeline step {final_name!r}."
        )
    return prefix


def _pipls_parameter_keys(prefix: str) -> tuple[str, str]:
    separator = "__" if prefix else ""
    return (
        f"{prefix}{separator}n_components",
        f"{prefix}{separator}predictor_rank",
    )


def _pipls_parameter_key(prefix: str, parameter: str) -> str:
    separator = "__" if prefix else ""
    return f"{prefix}{separator}{parameter}"


def _extract_fitted_pipls(estimator: Any, prefix: str) -> PiPLSRegression:
    if isinstance(estimator, PiPLSRegression):
        if prefix != "":
            raise ValueError("A direct fitted PiPLSRegression requires an empty prefix.")
        return estimator
    if isinstance(estimator, Pipeline):
        value = estimator.named_steps.get(prefix)
        if isinstance(value, PiPLSRegression) and estimator.steps[-1][0] == prefix:
            return value
    raise ValueError(
        f"Fitted parameter prefix {prefix!r} no longer locates the final "
        "PiPLSRegression step."
    )


def _fold_safe_feature_limit(
    *,
    template: Any,
    prefix: str,
    X: ArrayLike,
    y: ArrayLike,
    splits: tuple[tuple[IntArray, IntArray], ...],
) -> int:
    if prefix == "" and isinstance(template, PiPLSRegression):
        return int(np.asarray(X).shape[1])
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
    cache: dict[tuple[int, int], _PiPLSCandidateResult],
    template: Any,
    n_components_key: str,
    predictor_rank_key: str,
    scorer: Scorer | None,
    scoring: Scoring,
    X: ArrayLike,
    y: ArrayLike,
    splits: tuple[tuple[IntArray, IntArray], ...],
    n_jobs: int | None,
) -> tuple[int, ...]:
    candidates = tuple(
        _PiPLSCandidate(n_components=h, predictor_rank=r) for h, r in pairs
    )
    evaluated = _evaluate_candidate_batch(
        candidates=candidates,
        cache=cache,
        template=template,
        n_components_key=n_components_key,
        predictor_rank_key=predictor_rank_key,
        scorer=scorer,
        use_default_scoring=_uses_default_path_scoring(scoring),
        X=X,
        y=y,
        splits=splits,
        n_jobs=n_jobs,
    )
    return tuple(candidate.predictor_rank for candidate in evaluated)


def _adaptive_path_search(
    *,
    n_components: int,
    allowed_ranks: IntArray,
    cache: dict[tuple[int, int], _PiPLSCandidateResult],
    template: Any,
    n_components_key: str,
    predictor_rank_key: str,
    scorer: Scorer | None,
    scoring: Scoring,
    X: ArrayLike,
    y: ArrayLike,
    splits: tuple[tuple[IntArray, IntArray], ...],
    n_jobs: int | None,
) -> tuple[tuple[int, ...], ...]:
    def evaluate(ranks: IntArray) -> IntArray:
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
        return np.asarray(evaluated, dtype=np.intp)

    def evaluated_scores() -> tuple[IntArray, FloatArray]:
        ranks = np.asarray(
            sorted(r for h, r in cache if h == n_components),
            dtype=np.intp,
        )
        scores = np.asarray(
            [
                np.mean(cache[(n_components, int(rank))].split_scores)
                for rank in ranks
            ],
            dtype=np.float64,
        )
        return ranks, scores

    search = _search_predictor_ranks(
        allowed_ranks=allowed_ranks,
        search_method="auto",
        evaluate=evaluate,
        evaluated_scores=evaluated_scores,
    )
    return tuple(
        tuple(int(rank) for rank in batch) for batch in search.history
    )


def _uses_default_path_scoring(scoring: Scoring) -> bool:
    """Return whether the shared default response-standardized scorer is requested."""

    return isinstance(scoring, str) and scoring == _DEFAULT_SCORING


def _path_cv_results(
    *,
    cache: dict[tuple[int, int], _PiPLSCandidateResult],
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
    split_fit_times = np.vstack([cache[pair].split_fit_times for pair in evaluated_pairs])
    split_score_times = np.vstack([cache[pair].split_score_times for pair in evaluated_pairs])
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
        "mean_fit_time": np.mean(split_fit_times, axis=1),
        "std_fit_time": np.std(split_fit_times, axis=1),
        "mean_score_time": np.mean(split_score_times, axis=1),
        "std_score_time": np.std(split_score_times, axis=1),
        "mean_response_standardized_mse": np.mean(split_mse, axis=1),
        "std_response_standardized_mse": np.std(split_mse, axis=1),
    }
    results["rank_test_score"] = _rank_test_scores(mean_scores)
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


def _resolve_path_scorer(scoring: Scoring, estimator: Any) -> Scorer | None:
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
    if cv is None:
        return
    if isinstance(cv, (float, np.floating, str, bytes)):
        raise ValueError("cv must be an integer at least 2, a splitter, or an iterable of splits.")
