"""Pipeline-aware cross-validated Pi-PLS model selection."""

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

from ._core import _PredictorRankInfeasibleError
from ._cv_engine import (
    CandidateCache,
    _evaluate_candidate_batch,
    _fit_with_ignored_warnings,
    _ordered_oof_predictions,
    _PiPLSCandidate,
)
from ._sklearn_compat import _validate_estimator_data
from .component_path import (
    PiPLSComponentPath,
    PiPLSPredictorRankProfile,
    PredictorRankPolicy,
)
from .exceptions import StatisticalSupportWarning
from .metrics import neg_response_standardized_mean_squared_error
from .model_selection import (
    CVSplit,
    _as_positive_float,
    _is_leave_one_out_splits,
    _materialize_cv_splits,
    _max_predictor_rank,
    _pooled_oof_r2,
    _rank_test_scores,
    _search_predictor_ranks,
    _tied_score_mask,
    _validate_positive_int,
    _validate_singleton_fold_scoring,
)
from .regression import PiPLSRegression, _clear_fitted_state
from .validation import PiPLSValidationReport

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.intp]
Scorer = Callable[[Any, ArrayLike, ArrayLike], float]
Scoring = str | Scorer | None
SearchMethod = Literal["optimal", "auto"]
SelectionRule = Literal["best_score", "one_standard_error"]
ComponentValues = Sequence[int] | Literal["all"]
PredictorRankValues = Sequence[int] | Literal["max"] | None
_DEFAULT_SCORING_NAME = "neg_response_standardized_mean_squared_error"
_MIN_TRUSTED_SAMPLES_PER_PREDICTOR_RANK = 5.0
_CONTROLLED_FIT_WARNING_CATEGORIES = (StatisticalSupportWarning,)


def _default_pipls_template() -> PiPLSRegression:
    """Return the direct path template with a valid construction seed pair."""

    return PiPLSRegression(n_components=1, predictor_rank=1)


def _estimator_supports(method_name: str) -> Callable[[Any], bool]:
    def check(search: Any) -> bool:
        if not bool(search.refit):
            return False
        if hasattr(search, "selected_params_") and not hasattr(
            search, "selected_estimator_"
        ):
            return False
        estimator = (
            search.selected_estimator_
            if hasattr(search, "selected_estimator_")
            else (_default_pipls_template() if search.estimator is None else search.estimator)
        )
        return hasattr(estimator, method_name)

    return check


class PiPLSSearchCV(
    TransformerMixin,  # type: ignore[misc]
    RegressorMixin,  # type: ignore[misc]
    MultiOutputMixin,  # type: ignore[misc]
    MetaEstimatorMixin,  # type: ignore[misc]
    BaseEstimator,  # type: ignore[misc]
    auto_wrap_output_keys=None,  # type: ignore[call-arg]
):
    r"""Cross-validated search over the admissible Pi-PLS rank path.

    Every candidate is a fixed-rank :class:`pipls.PiPLSRegression` clone fitted
    independently inside each training fold. A declared selection rule chooses
    one stored path row, which is optionally refitted on all supplied data. The
    default ``search_method="auto"`` applies
    a deterministic logarithmic coarse-to-fine predictor-rank search separately
    for each component count.

    Parameters
    ----------
    estimator : PiPLSRegression, sklearn.pipeline.Pipeline or None, default=None
        Direct Pi-PLS estimator or pipeline whose final step is
        :class:`pipls.PiPLSRegression`. ``None`` creates a direct template with
        the valid construction seed pair ``(n_components=1, predictor_rank=1)``.
        Path preflight and candidate evaluation replace both values before fitting.
    n_components_values : sequence of int or "all", default="all"
        Positive component counts to evaluate. ``"all"`` uses every value from
        one through ``min(n_targets_, max_predictor_rank_)``.
    predictor_rank_values : sequence of int, "max" or None, default=None
        Admissible predictor ranks. ``None`` makes every rank from one through
        ``max_predictor_rank_`` available; ``search_method`` determines which
        are evaluated. A one-element sequence fixes one rank, a longer sequence
        defines an explicit set, and ``"max"`` uses ``max_predictor_rank_`` for
        every component count.
    max_predictor_rank : int or "rule", default="rule"
        ``"rule"`` applies the total-sample support rule together with
        dimensional and verified numerical-rank caps from every training fold.
        A positive integer imposes an additional upper bound but bypasses only
        the support rule.
    search_method : {"auto", "optimal"}, default="auto"
        ``"optimal"`` evaluates every admissible pair. ``"auto"`` uses the
        deterministic adaptive search and may skip pairs.
    samples_per_predictor_rank : float, default=5
        Positive support parameter $c$ for ``max_predictor_rank="rule"``. Values
        below five issue :class:`pipls.StatisticalSupportWarning`.
    cv : int, splitter, iterable or None, default=5
        Cross-validation specification. ``None`` requests the standard five-fold
        regression split.
    scoring : str, callable or None, default="neg_response_standardized_mean_squared_error"
        Scikit-learn scorer name, scorer callable, or ``None`` to use estimator
        ``score``. The package-specific default name resolves to
        :func:`pipls.metrics.neg_response_standardized_mean_squared_error`.
    selection_rule : {"best_score", "one_standard_error"}, default="best_score"
        Rule used to choose the final component-path row. ``"best_score"`` uses
        the globally best evaluated pair under ``scoring``. ``"one_standard_error"``
        uses :meth:`PiPLSComponentPath.one_standard_error_result`, retaining the
        predictor rank already selected conditionally for that component count.
    refit : bool, default=False
        Whether to refit the row chosen by ``selection_rule`` on all supplied
        data. The default leaves path evaluation and final fixed-model fitting
        as separate steps. Delegated prediction, transformation, scoring, and
        feature-name methods require ``refit=True``.
    n_jobs : int or None, default=None
        Joblib parallelism across candidate pairs within each evaluation batch.
    return_oof_predictions : bool, default=False
        Whether to fit the selected fixed parameterization on every training fold
        and retain row-ordered validation predictions. Repeated predictions are
        averaged and rows never validated are marked with NaN.

    Attributes
    ----------
    n_features_in_ : int
        Number of predictor columns seen during fitting.
    feature_names_in_ : ndarray of shape (n_features_in_,)
        Predictor names seen during fitting. Defined only when all input feature
        names are strings.
    n_targets_ : int
        Number of response columns seen during fitting.
    n_splits_ : int
        Number of materialized cross-validation splits.
    cv_n_train_min_ : int
        Smallest training-fold size.
    max_predictor_rank_ : int
        Effective predictor-rank upper bound after support, dimensional, and
        verified fold-numerical-rank constraints.
    path_search_exhaustive_ : bool
        Whether every admissible pair was evaluated.
    scorer_ : callable
        Validated scikit-learn scorer used during fitting.
    cv_results_ : dict of str to array-like
        Full candidate-level results. It includes parameter pairs, split scores,
        response-standardized MSE values, timing summaries, and minimum score
        ranks formed with the same tolerant comparison used for selection.
    component_path_ : PiPLSComponentPath
        Immutable concise view with one conditionally selected predictor-rank
        result per component count. Use :meth:`predictor_rank_profile` for the
        evaluated rank candidates at one component count.
    best_index_ : int
        Row of ``cv_results_`` selected by maximum mean test score, with smaller
        component count and predictor rank used as deterministic tie-breakers.
    best_score_ : float
        Mean cross-validation score at ``best_index_``.
    best_n_components_ : int
        Selected component count.
    best_predictor_rank_ : int
        Selected predictor rank.
    best_params_ : dict of str to int
        Parameters required to configure the supplied estimator or pipeline.
    selected_result_ : PiPLSComponentResult
        Immutable component-path row chosen by ``selection_rule``.
    selected_params_ : dict of str to int
        Parameters required to configure the supplied estimator or pipeline for
        ``selected_result_``.
    validation_report_ : PiPLSValidationReport
        Immutable summary of ``selected_result_`` and optional ordered OOF
        predictions for that fixed parameterization.
    selected_estimator_ : estimator
        Estimator refitted on all data with ``selected_params_``. Defined only
        when ``refit=True``.
    selected_pipls_ : PiPLSRegression
        Fitted terminal Pi-PLS estimator. Defined only when ``refit=True``.
    best_estimator_ : estimator
        Compatibility alias of ``selected_estimator_``. Defined only when
        ``refit=True`` and ``selection_rule="best_score"``.
    best_pipls_ : PiPLSRegression
        Compatibility alias of ``selected_pipls_``. Defined only when
        ``refit=True`` and ``selection_rule="best_score"``.
    refit_time_ : float
        Selected full-data refit time in seconds. Defined only when ``refit=True``.
    """

    def __init__(
        self,
        estimator: Any | None = None,
        *,
        n_components_values: ComponentValues = "all",
        predictor_rank_values: PredictorRankValues = None,
        max_predictor_rank: int | Literal["rule"] = "rule",
        search_method: SearchMethod = "auto",
        samples_per_predictor_rank: float = 5.0,
        cv: object = 5,
        scoring: Scoring = _DEFAULT_SCORING_NAME,
        selection_rule: SelectionRule = "best_score",
        refit: bool = False,
        n_jobs: int | None = None,
        return_oof_predictions: bool = False,
    ) -> None:
        self.estimator = estimator
        self.n_components_values = n_components_values
        self.predictor_rank_values = predictor_rank_values
        self.max_predictor_rank = max_predictor_rank
        self.search_method = search_method
        self.samples_per_predictor_rank = samples_per_predictor_rank
        self.cv = cv
        self.scoring = scoring
        self.selection_rule = selection_rule
        self.refit = refit
        self.n_jobs = n_jobs
        self.return_oof_predictions = return_oof_predictions

    def fit(
        self,
        X: ArrayLike,
        y: ArrayLike,
        *,
        groups: ArrayLike | None = None,
    ) -> PiPLSSearchCV:
        """Evaluate the path and optionally refit the selected path row.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Predictor matrix.
        y : array-like of shape (n_samples,) or (n_samples, n_targets)
            Response vector or matrix.
        groups : array-like of shape (n_samples,), optional
            Group labels passed to a group-aware splitter.

        Returns
        -------
        self : PiPLSSearchCV
            Fitted search object.
        """

        _clear_fitted_state(self)
        try:
            return self._fit(X, y, groups=groups)
        except Exception:
            _clear_fitted_state(self)
            raise

    def _fit(
        self,
        X: ArrayLike,
        y: ArrayLike,
        *,
        groups: ArrayLike | None,
    ) -> PiPLSSearchCV:
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
            copy=True,
        )
        X_checked, y_checked = cast(tuple[Any, Any], validated)
        X_array = np.asarray(X_checked, dtype=np.float64)
        y_array = np.array(y_checked, dtype=np.float64, copy=True)
        y_2d = y_array.reshape(-1, 1) if y_array.ndim == 1 else y_array
        self.n_targets_ = int(y_2d.shape[1])
        X_indexable, y_indexable = indexable(X, y)

        template = _default_pipls_template() if self.estimator is None else self.estimator
        pipls_param_prefix = _resolve_pipls_param_prefix(template)
        n_components_key, predictor_rank_key = _pipls_parameter_keys(
            pipls_param_prefix
        )
        template = clone(template)
        materialized = _materialize_cv_splits(self.cv, X_array, y_array, groups=groups)
        _validate_singleton_fold_scoring(self.scoring, materialized.splits)
        self.n_splits_ = len(materialized.splits)
        self.cv_n_train_min_ = materialized.n_train_min
        if self.selection_rule == "one_standard_error" and self.n_splits_ < 2:
            raise ValueError(
                'selection_rule="one_standard_error" requires at least two '
                "validation splits."
            )

        fold_feature_limit, fold_numerical_rank_limit = _fold_predictor_limits(
            template=template,
            X=X_indexable,
            y=y_indexable,
            splits=materialized.splits,
        )
        algebraic_limit = min(fold_feature_limit, materialized.n_train_min - 1)
        if self.max_predictor_rank == "rule":
            support_limit = _max_predictor_rank(
                n_features=fold_feature_limit,
                n_samples=int(X_array.shape[0]),
                n_train_min=materialized.n_train_min,
                samples_per_predictor_rank=self.samples_per_predictor_rank,
            )
            self.max_predictor_rank_ = min(
                support_limit,
                fold_numerical_rank_limit,
            )
        else:
            self.max_predictor_rank_ = min(
                int(self.max_predictor_rank),
                algebraic_limit,
                fold_numerical_rank_limit,
            )

        h_limit = min(self.n_targets_, self.max_predictor_rank_)
        component_values = _validated_component_values(
            self.n_components_values,
            upper=h_limit,
        )
        predictor_rank_policy = _predictor_rank_policy(self.predictor_rank_values)
        if isinstance(self.predictor_rank_values, str):
            predictor_rank_values = np.asarray(
                [self.max_predictor_rank_], dtype=np.intp
            )
        else:
            predictor_rank_values = _validated_integer_values(
                self.predictor_rank_values,
                name="predictor_rank_values",
                lower=1,
                upper=self.max_predictor_rank_,
            )
        admissible = tuple(
            (int(h), int(r))
            for h in component_values
            for r in predictor_rank_values
            if h <= r
        )
        if not admissible:
            raise ValueError(
                "The supplied n_components_values and predictor_rank_values do not "
                "contain an admissible pair satisfying n_components <= predictor_rank."
            )
        missing_components = [
            int(h)
            for h in component_values
            if not any(hh == int(h) for hh, _ in admissible)
        ]
        if missing_components:
            missing = ", ".join(map(str, missing_components))
            raise ValueError(
                "Every n_components value must have at least one admissible predictor "
                f"rank; missing ranks for {missing}."
            )
        scorer = _resolve_path_scorer(self.scoring, template)
        self.scorer_ = scorer
        cache: CandidateCache = {}
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
        else:
            for h_value in component_values:
                h = int(h_value)
                allowed = np.asarray(
                    [r for hh, r in admissible if hh == h],
                    dtype=np.intp,
                )
                _adaptive_path_search(
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
        self.path_search_exhaustive_ = len(evaluated_pairs) == len(admissible)

        self.cv_results_ = _path_cv_results(
            cache=cache,
            evaluated_pairs=evaluated_pairs,
            n_components_key=n_components_key,
            predictor_rank_key=predictor_rank_key,
        )
        self.best_index_ = _select_best_index(self.cv_results_)
        self.best_score_ = float(self.cv_results_["mean_test_score"][self.best_index_])
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

        conditional_indices: list[int] = []
        for h_value in component_values:
            indices = np.flatnonzero(self.cv_results_["n_components"] == int(h_value))
            if indices.size:
                conditional_indices.append(
                    _select_best_index(self.cv_results_, indices)
                )

        self.component_path_ = _component_path(
            results=self.cv_results_,
            conditional_indices=np.asarray(conditional_indices, dtype=np.intp),
            predictor_rank_policy=predictor_rank_policy,
            n_splits=self.n_splits_,
        )
        if self.selection_rule == "best_score":
            self.selected_result_ = self.component_path_.for_n_components(
                self.best_n_components_
            )
        else:
            self.selected_result_ = self.component_path_.one_standard_error_result()
        self.selected_params_ = {
            n_components_key: self.selected_result_.n_components,
            predictor_rank_key: self.selected_result_.predictor_rank,
        }

        predictions: FloatArray | None = None
        counts: IntArray | None = None
        pooled_r2: float | None = None
        if self.return_oof_predictions:
            oof = _ordered_oof_predictions(
                candidate=_PiPLSCandidate(
                    n_components=self.selected_result_.n_components,
                    predictor_rank=self.selected_result_.predictor_rank,
                ),
                template=template,
                n_components_key=n_components_key,
                predictor_rank_key=predictor_rank_key,
                X=X_indexable,
                y=y_indexable,
                splits=materialized.splits,
                n_jobs=self.n_jobs,
                ignored_warning_categories=_CONTROLLED_FIT_WARNING_CATEGORIES,
            )
            oof_predictions, counts = oof
            predictions = (
                oof_predictions[:, 0] if y_array.ndim == 1 else oof_predictions
            )
            pooled_r2 = _pooled_oof_r2(y_indexable, predictions, counts)

        self.validation_report_ = PiPLSValidationReport(
            n_components=self.selected_result_.n_components,
            predictor_rank=self.selected_result_.predictor_rank,
            n_splits=self.n_splits_,
            mean_test_score=self.selected_result_.mean_test_score,
            mean_response_standardized_mse=self.selected_result_.cv_mse_mean,
            estimate_kind="selection-conditioned",
            is_leave_one_out=_is_leave_one_out_splits(
                materialized.splits,
                n_samples=int(y_array.shape[0]),
            ),
            oof_predictions=predictions,
            oof_prediction_counts=counts,
            pooled_oof_r2=pooled_r2,
        )
        if self.refit:
            refit_started = perf_counter()
            self.selected_estimator_ = clone(template).set_params(**self.selected_params_)
            _fit_controlled_estimator(
                self.selected_estimator_,
                X_indexable,
                y_indexable,
            )
            self.refit_time_ = perf_counter() - refit_started
            self.selected_pipls_ = _extract_fitted_pipls(
                self.selected_estimator_,
                pipls_param_prefix,
            )
            if self.selection_rule == "best_score":
                self.best_estimator_ = self.selected_estimator_
                self.best_pipls_ = self.selected_pipls_
        return self

    def predictor_rank_profile(
        self,
        n_components: int,
    ) -> PiPLSPredictorRankProfile:
        """Return evaluated predictor-rank results for one component count.

        Rows are sorted by ascending predictor rank and include only candidates
        actually evaluated by the fitted search. The selected scalar result
        maximizes the configured mean test score, with the fitted conditional
        tie-breaking rule. Under the default scorer, this is equivalent to
        minimizing mean response-standardized CV-MSE.

        Parameters
        ----------
        n_components : int
            Evaluated component count whose predictor-rank profile is requested.

        Returns
        -------
        PiPLSPredictorRankProfile
            Frozen result containing aligned read-only candidate arrays and the
            conditionally selected scalar result.

        Raises
        ------
        sklearn.exceptions.NotFittedError
            If the search has not been fitted.
        ValueError
            If ``n_components`` is not an integer or was not evaluated.
        """

        check_is_fitted(self, attributes=["cv_results_", "component_path_", "n_splits_"])
        selected = self.component_path_.for_n_components(n_components)
        rows = np.flatnonzero(
            cast(IntArray, self.cv_results_["n_components"]) == selected.n_components
        )
        ranks = cast(IntArray, self.cv_results_["predictor_rank"])[rows]
        order = np.argsort(ranks)
        indices = rows[order]
        return PiPLSPredictorRankProfile(
            n_components=selected.n_components,
            predictor_rank=cast(IntArray, self.cv_results_["predictor_rank"])[indices],
            mean_test_score=cast(FloatArray, self.cv_results_["mean_test_score"])[indices],
            cv_mse_mean=cast(
                FloatArray,
                self.cv_results_["mean_response_standardized_mse"],
            )[indices],
            cv_mse_fold_sd=cast(
                FloatArray,
                self.cv_results_["std_response_standardized_mse"],
            )[indices],
            n_splits=self.n_splits_,
            selected=selected,
        )

    @available_if(_estimator_supports("predict"))  # type: ignore[untyped-decorator]
    def predict(self, X: ArrayLike, copy: bool = True) -> FloatArray:
        """Predict with the refitted selected estimator.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Predictor matrix.
        copy : bool, default=True
            Whether validation may copy ``X`` for a direct Pi-PLS estimator.

        Returns
        -------
        y_pred : ndarray
            Predictions from ``selected_estimator_``.

        Notes
        -----
        This method is available only when ``refit=True`` and the estimator
        template supports prediction.
        """

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
        """Transform with the refitted selected estimator.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Predictor matrix.
        y : array-like, optional
            Responses to transform with a direct ``PiPLSRegression``. Composite
            estimators support predictor transformation only.
        copy : bool, default=True
            Whether validation may copy arrays for a direct Pi-PLS estimator.

        Returns
        -------
        transformed : ndarray or tuple of ndarray
            Output of the selected estimator's transformation.

        Notes
        -----
        This method is available only when ``refit=True`` and the estimator
        template supports transformation.
        """

        estimator = self._refitted_estimator()
        if isinstance(estimator, PiPLSRegression):
            return estimator.transform(X, y, copy=copy)
        if y is not None:
            raise ValueError(
                "transform(X, y) is available when the refitted estimator is a direct "
                "PiPLSRegression. For composite estimators, use selected_pipls_ with data "
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
        """Fit the path search and transform with the selected estimator.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Predictor matrix.
        y : array-like of shape (n_samples,) or (n_samples, n_targets)
            Response vector or matrix. ``y`` is required.
        groups : array-like of shape (n_samples,), optional
            Group labels passed to a group-aware splitter.
        **fit_params : dict
            Additional fit parameters are not supported and raise ``TypeError``.

        Returns
        -------
        transformed : ndarray or tuple of ndarray
            Transformation of the fitted data by ``selected_estimator_``.

        Notes
        -----
        This method is available only when ``refit=True`` and the estimator
        template supports transformation.
        """

        if fit_params:
            names = ", ".join(sorted(fit_params))
            raise TypeError(f"Unexpected fit parameters: {names}.")
        if y is None:
            raise ValueError("y is required to fit PiPLSSearchCV.")
        self.fit(X, y, groups=groups)
        if isinstance(self.selected_estimator_, PiPLSRegression):
            return cast(tuple[FloatArray, FloatArray], self.transform(X, y))
        return cast(Any, self.transform(X))

    @available_if(_estimator_supports("inverse_transform"))  # type: ignore[untyped-decorator]
    def inverse_transform(
        self,
        X: ArrayLike,
        y: ArrayLike | None = None,
    ) -> Any:
        """Reconstruct data through the refitted selected estimator.

        Parameters
        ----------
        X : array-like
            Transformed predictor representation.
        y : array-like, optional
            Response scores for a direct ``PiPLSRegression``. Composite
            estimators reconstruct predictors only.

        Returns
        -------
        reconstructed : ndarray or tuple of ndarray
            Output of the selected estimator's inverse transformation.

        Notes
        -----
        This method is available only when ``refit=True`` and the estimator
        template supports inverse transformation.
        """

        estimator = self._refitted_estimator()
        if isinstance(estimator, PiPLSRegression):
            return estimator.inverse_transform(X, y)
        if y is not None:
            raise ValueError(
                "inverse_transform(X, y) is available when the refitted estimator "
                "is a direct PiPLSRegression. Composite estimators reconstruct X only."
            )
        return estimator.inverse_transform(X)

    @available_if(_estimator_supports("score"))  # type: ignore[untyped-decorator]
    def score(
        self,
        X: ArrayLike,
        y: ArrayLike,
        sample_weight: ArrayLike | None = None,
    ) -> float:
        r"""Return the selected estimator's uniformly averaged $R^2$.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Predictor matrix.
        y : array-like of shape (n_samples,) or (n_samples, n_targets)
            Observed responses.
        sample_weight : array-like of shape (n_samples,), optional
            Sample weights forwarded to the selected estimator.

        Returns
        -------
        score : float
            Score returned by ``selected_estimator_``.

        Notes
        -----
        This method is available only when ``refit=True`` and the estimator
        template supports scoring.
        """

        estimator = self._refitted_estimator()
        if sample_weight is None:
            return float(estimator.score(X, y))
        return float(estimator.score(X, y, sample_weight=sample_weight))

    @available_if(_estimator_supports("get_feature_names_out"))  # type: ignore[untyped-decorator]
    def get_feature_names_out(
        self,
        input_features: ArrayLike | None = None,
    ) -> NDArray[np.object_]:
        """Return names for the selected transformed predictor features.

        Parameters
        ----------
        input_features : array-like of str, optional
            Input feature names validated by the selected estimator.

        Returns
        -------
        feature_names_out : ndarray of str
            Names returned by ``selected_estimator_`` or its fitted terminal
            ``PiPLSRegression``.

        Notes
        -----
        This method is available only when ``refit=True`` and the estimator
        template supports output feature names.
        """

        estimator = self._refitted_estimator()
        method = getattr(estimator, "get_feature_names_out", None)
        if method is not None:
            return cast(NDArray[np.object_], method(input_features))
        return cast(
            NDArray[np.object_],
            self.selected_pipls_.get_feature_names_out(input_features),
        )

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
        check_is_fitted(self, attributes=["selected_params_"])
        if not hasattr(self, "selected_estimator_"):
            raise AttributeError(
                "PiPLSSearchCV was fitted with refit=False; predict, transform, and score "
                "require refit=True."
            )
        return self.selected_estimator_

    def _validate_constructor_parameters(self) -> None:
        if self.estimator is not None:
            _validate_supported_estimator(self.estimator)
        if self.search_method not in ("optimal", "auto"):
            raise ValueError('search_method must be "optimal" or "auto".')
        if self.selection_rule not in ("best_score", "one_standard_error"):
            raise ValueError(
                'selection_rule must be "best_score" or "one_standard_error".'
            )
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
                "This permits fewer than five supplied samples per retained predictor-rank "
                "direction, so the resulting rank bound may not have sufficient statistical "
                "support to be trusted without external validation.",
                StatisticalSupportWarning,
                stacklevel=3,
            )
        if self.max_predictor_rank != "rule":
            _validate_positive_int(self.max_predictor_rank, name="max_predictor_rank")
        _validate_component_values(self.n_components_values)
        _validate_predictor_rank_values(self.predictor_rank_values)
        _validate_n_jobs(self.n_jobs)
        _validate_cv(self.cv)
        scorer_template = (
            _default_pipls_template() if self.estimator is None else self.estimator
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


def _resolve_pipls_param_prefix(template: Any) -> str:
    """Return the unique supported Pi-PLS parameter prefix."""

    _validate_supported_estimator(template)
    if isinstance(template, PiPLSRegression):
        return ""

    assert isinstance(template, Pipeline)
    return cast(str, template.steps[-1][0])


def _pipls_parameter_keys(prefix: str) -> tuple[str, str]:
    separator = "__" if prefix else ""
    return (
        f"{prefix}{separator}n_components",
        f"{prefix}{separator}predictor_rank",
    )


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


def _fold_predictor_limits(
    *,
    template: Any,
    X: ArrayLike,
    y: ArrayLike,
    splits: tuple[CVSplit, ...],
) -> tuple[int, int]:
    """Return minimum transformed feature count and verified rank across folds."""

    feature_counts: list[int] = []
    numerical_ranks: list[int] = []
    global_random_state = np.random.get_state()
    try:
        for split_index, (train, _) in enumerate(splits):
            X_train = _safe_indexing(X, train)
            y_train = _safe_indexing(y, train)
            final_estimator, X_transformed = _fold_final_estimator_and_predictors(
                template=template,
                X=X_train,
                y=y_train,
            )
            transformed_shape = np.shape(X_transformed)
            if len(transformed_shape) != 2:
                raise ValueError(
                    "Pipeline preprocessing must produce a two-dimensional predictor "
                    f"matrix; got shape={transformed_shape}."
                )
            n_samples, n_features = transformed_shape
            feature_counts.append(int(n_features))
            algebraic_limit = min(int(n_features), int(n_samples) - 1)
            if algebraic_limit < 1:
                raise ValueError(
                    "No positive predictor rank is feasible after preprocessing in "
                    f"training split {split_index}."
                )
            probe = final_estimator.set_params(
                n_components=1,
                predictor_rank=algebraic_limit,
            )
            try:
                _fit_controlled_estimator(probe, X_transformed, y_train)
            except _PredictorRankInfeasibleError as error:
                verified_rank = error.verified_rank
            else:
                verified_rank = algebraic_limit
            if verified_rank < 1:
                raise ValueError(
                    "No positive predictor rank is numerically feasible after "
                    f"preprocessing in training split {split_index}."
                )
            numerical_ranks.append(verified_rank)
    finally:
        np.random.set_state(global_random_state)
    return min(feature_counts), min(numerical_ranks)


def _fold_final_estimator_and_predictors(
    *,
    template: Any,
    X: ArrayLike,
    y: ArrayLike,
) -> tuple[PiPLSRegression, Any]:
    """Fit fold-local preprocessing and return its final Pi-PLS template and X."""

    if isinstance(template, PiPLSRegression):
        return clone(template), X

    assert isinstance(template, Pipeline)
    final_estimator = cast(PiPLSRegression, clone(template.steps[-1][1]))
    if len(template.steps) == 1:
        return final_estimator, X
    preprocessor = clone(template[:-1])
    X_transformed = preprocessor.fit_transform(X, y)
    return final_estimator, X_transformed


def _evaluate_path_batch(
    *,
    pairs: Iterable[tuple[int, int]],
    cache: CandidateCache,
    template: Any,
    n_components_key: str,
    predictor_rank_key: str,
    scorer: Scorer | None,
    scoring: Scoring,
    X: ArrayLike,
    y: ArrayLike,
    splits: tuple[CVSplit, ...],
    n_jobs: int | None,
) -> None:
    _evaluate_candidate_batch(
        candidates=(
            _PiPLSCandidate(n_components=h, predictor_rank=r) for h, r in pairs
        ),
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
        ignored_warning_categories=_CONTROLLED_FIT_WARNING_CATEGORIES,
    )


def _fit_controlled_estimator(estimator: Any, X: ArrayLike, y: ArrayLike) -> None:
    """Fit one path-owned estimator while suppressing only the support diagnostic."""

    _fit_with_ignored_warnings(
        estimator,
        X,
        y,
        ignored_warning_categories=_CONTROLLED_FIT_WARNING_CATEGORIES,
    )


def _adaptive_path_search(
    *,
    n_components: int,
    allowed_ranks: IntArray,
    cache: CandidateCache,
    template: Any,
    n_components_key: str,
    predictor_rank_key: str,
    scorer: Scorer | None,
    scoring: Scoring,
    X: ArrayLike,
    y: ArrayLike,
    splits: tuple[CVSplit, ...],
    n_jobs: int | None,
) -> None:
    def evaluate(ranks: IntArray) -> None:
        _evaluate_path_batch(
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

    _search_predictor_ranks(
        allowed_ranks=allowed_ranks,
        search_method="auto",
        evaluate=evaluate,
        evaluated_scores=evaluated_scores,
    )


def _uses_default_path_scoring(scoring: Scoring) -> bool:
    """Return whether the public default response-standardized scorer is requested."""

    return isinstance(scoring, str) and scoring == _DEFAULT_SCORING_NAME


def _path_cv_results(
    *,
    cache: CandidateCache,
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


def _component_path(
    *,
    results: dict[str, Any],
    conditional_indices: IntArray,
    predictor_rank_policy: PredictorRankPolicy,
    n_splits: int,
) -> PiPLSComponentPath:
    """Return one conditionally selected predictor-rank result per component count."""

    n_rows = int(conditional_indices.size)
    return PiPLSComponentPath(
        n_components=cast(IntArray, results["n_components"])[conditional_indices],
        predictor_rank=cast(IntArray, results["predictor_rank"])[conditional_indices],
        predictor_rank_policy=np.full(n_rows, predictor_rank_policy, dtype=object),
        mean_test_score=cast(FloatArray, results["mean_test_score"])[conditional_indices],
        cv_mse_mean=cast(
            FloatArray,
            results["mean_response_standardized_mse"],
        )[conditional_indices],
        cv_mse_fold_sd=cast(
            FloatArray,
            results["std_response_standardized_mse"],
        )[conditional_indices],
        n_splits=np.full(n_rows, n_splits, dtype=np.intp),
    )


def _select_best_index(
    results: dict[str, Any],
    indices: IntArray | None = None,
) -> int:
    all_scores = cast(FloatArray, results["mean_test_score"])
    candidate_indices = (
        np.arange(all_scores.size, dtype=np.intp) if indices is None else indices
    )
    scores = all_scores[candidate_indices]
    maximum = float(np.max(scores))
    tied = candidate_indices[_tied_score_mask(scores, maximum)]
    n_components = cast(IntArray, results["n_components"])[tied]
    predictor_rank = cast(IntArray, results["predictor_rank"])[tied]
    return int(tied[np.lexsort((predictor_rank, n_components))[0]])


def _validated_component_values(values: ComponentValues, *, upper: int) -> IntArray:
    if isinstance(values, str) and values == "all":
        return np.arange(1, upper + 1, dtype=np.intp)
    return _validated_integer_values(
        values,
        name="n_components_values",
        lower=1,
        upper=upper,
    )


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


def _validate_component_values(values: object) -> None:
    if values is None:
        raise ValueError(
            'n_components_values must be "all" or a sequence of positive integers.'
        )
    if isinstance(values, str):
        if values == "all":
            return
        raise ValueError(
            'n_components_values must be "all" or a sequence of positive integers.'
        )
    _validate_integer_sequence(values, name="n_components_values")


def _validate_predictor_rank_values(values: object) -> None:
    if values is None or (isinstance(values, str) and values == "max"):
        return
    if isinstance(values, (str, bytes)):
        raise ValueError(
            'predictor_rank_values must be None, "max", or a sequence of positive integers.'
        )
    _validate_integer_sequence(values, name="predictor_rank_values")


def _validate_integer_sequence(values: object, *, name: str) -> None:
    try:
        sequence = tuple(cast(Iterable[object], values))
    except TypeError as error:
        raise ValueError(f"{name} must be a sequence of positive integers.") from error
    if not sequence:
        raise ValueError(f"{name} must not be empty.")
    for value in sequence:
        _validate_positive_int(value, name=name)


def _predictor_rank_policy(values: PredictorRankValues) -> PredictorRankPolicy:
    """Describe how predictor rank is supplied for the component path."""

    if isinstance(values, str):
        return "maximum"
    if values is None:
        return "optimized"
    return "fixed" if len(values) == 1 else "optimized"


def _resolve_path_scorer(scoring: Scoring, estimator: Any) -> Scorer:
    if isinstance(scoring, str) and scoring == _DEFAULT_SCORING_NAME:
        return neg_response_standardized_mean_squared_error
    if isinstance(scoring, str):
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
