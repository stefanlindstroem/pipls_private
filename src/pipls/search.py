"""Pipeline-aware cross-validated Pi-PLS model selection."""

from __future__ import annotations

import warnings
from collections.abc import Callable, Iterable, Sequence
from dataclasses import replace
from typing import Any, Literal, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.base import (
    BaseEstimator,
    MetaEstimatorMixin,
    MultiOutputMixin,
    clone,
)
from sklearn.metrics import check_scoring, get_scorer
from sklearn.pipeline import Pipeline
from sklearn.utils import _safe_indexing, indexable
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
    PiPLSComponentResult,
    PiPLSPredictorRankProfile,
    PredictorRankPolicy,
    SelectionRule,
)
from .exceptions import PredictorRankSupportWarning
from .metrics import neg_response_standardized_mse
from .model_selection import (
    CVSplit,
    _as_positive_float,
    _materialize_cv_splits,
    _max_predictor_rank,
    _pooled_oof_r2,
    _rank_test_scores,
    _search_predictor_ranks,
    _splits_are_leave_one_out,
    _tied_score_mask,
    _validate_positive_int,
    _validate_singleton_fold_scoring,
)
from .regression import PiPLSRegression, _clear_fitted_state
from .validation import PiPLSOOFReport, PiPLSValidationReport

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.intp]
Scorer = Callable[[Any, ArrayLike, ArrayLike], float]
Scoring = str | Scorer | None
SearchMethod = Literal["optimal", "auto"]
ComponentValues = Sequence[int] | Literal["all"]
PredictorRankValues = Sequence[int] | Literal["max"] | None
_DEFAULT_SCORING_NAME = "neg_response_standardized_mse"
_MIN_TRUSTED_SAMPLES_PER_PREDICTOR_RANK = 5.0
_CONTROLLED_FIT_WARNING_CATEGORIES = (PredictorRankSupportWarning,)


def _default_pipls_template() -> PiPLSRegression:
    """Return the direct path template with a valid construction seed pair."""

    return PiPLSRegression(n_components=1, predictor_rank=1)


def _component_result_at_count(
    path: PiPLSComponentPath,
    n_components: int,
) -> PiPLSComponentResult:
    """Return one stored component-path row by paired-mode count."""

    if isinstance(n_components, bool) or not isinstance(
        n_components,
        (int, np.integer),
    ):
        raise ValueError(
            "n_components must be an integer present in the fitted component path."
        )
    requested = int(n_components)
    index = int(np.searchsorted(path.n_components, requested))
    if index >= path.n_components.size or int(path.n_components[index]) != requested:
        available = ", ".join(str(int(value)) for value in path.n_components)
        raise ValueError(
            f"n_components={requested} was not evaluated. Available values are "
            f"[{available}]."
        )
    return path._result_at_index(index)


def _select_minimum_cv_mse(path: PiPLSComponentPath) -> PiPLSComponentResult:
    """Return the first stored path row with minimum mean CV-MSE."""

    return replace(
        path._result_at_index(int(np.argmin(path.cv_mse_mean))),
        rule="minimum_cv_mse",
    )


def _select_one_standard_error(path: PiPLSComponentPath) -> PiPLSComponentResult:
    """Return the smallest stored component count within one standard error."""

    reference = _select_minimum_cv_mse(path)
    threshold = reference.cv_mse_mean + reference.cv_mse_standard_error
    if not np.isfinite(threshold):
        raise ValueError("The one-standard-error threshold must be finite.")
    eligible = np.flatnonzero(path.cv_mse_mean <= threshold)
    return replace(
        path._result_at_index(int(eligible[0])),
        rule="one_standard_error",
        reference_minimum=reference,
    )


class PiPLSSearchCV(
    MultiOutputMixin,  # type: ignore[misc]
    MetaEstimatorMixin,  # type: ignore[misc]
    BaseEstimator,  # type: ignore[misc]
):
    r"""Cross-validated search over the admissible Pi-PLS rank path.

    Every candidate is a fixed-rank :class:`pipls.PiPLSRegression` clone fitted
    independently inside each training fold. Selection inspection, final
    full-data fitting, and selection-conditioned out-of-fold reporting are
    explicit post-fit :meth:`select`, :meth:`refit`,
    and :meth:`oof_report` operations. The default
    ``search_method="auto"`` applies
    a deterministic logarithmic coarse-to-fine predictor-rank search separately
    for each paired-mode count.

    Parameters
    ----------
    estimator : PiPLSRegression, sklearn.pipeline.Pipeline or None, default=None
        Direct Pi-PLS estimator or pipeline whose final step is
        :class:`pipls.PiPLSRegression`. Path preflight and candidate evaluation
        replace only ``n_components`` and ``predictor_rank``; settings such as
        ``scale``, ``copy``, ``svd_solver``, and ``random_state`` are retained
        from the cloned template. ``None`` creates a direct template with the
        valid construction seed pair ``(n_components=1, predictor_rank=1)`` and
        the ordinary :class:`pipls.PiPLSRegression` defaults for other settings.
    n_components_values : sequence of int or "all", default="all"
        Positive paired latent-mode counts to evaluate. ``"all"`` uses every value
        from
        one through ``min(n_targets_, max_predictor_rank_)``.
    predictor_rank_values : sequence of int, "max" or None, default=None
        Admissible predictor ranks. ``None`` makes every rank from one through
        ``max_predictor_rank_`` available; ``search_method`` determines which
        are evaluated. A one-element sequence fixes one rank, a longer sequence
        defines an explicit set, and ``"max"`` uses ``max_predictor_rank_`` for
        every paired-mode count.
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
        below five issue :class:`pipls.PredictorRankSupportWarning`.
    cv : int, splitter, iterable or None, default=5
        Cross-validation specification. ``None`` requests the standard five-fold
        regression split.
    scoring : str, callable or None, default="neg_response_standardized_mse"
        Scikit-learn scorer name, scorer callable, or ``None`` to use estimator
        ``score``. The package-specific default name resolves to
        :func:`pipls.metrics.neg_response_standardized_mse`.
    n_jobs : int or None, default=None
        Joblib parallelism across candidate pairs within each evaluation batch.

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
    search_is_exhaustive_ : bool
        Whether every admissible pair was evaluated.
    scorer_ : callable
        Validated scikit-learn scorer used during fitting.
    cv_results_ : dict of str to array-like
        Full candidate-level results. It includes parameter pairs, split scores,
        response-standardized MSE values, timing summaries, and minimum score
        ranks formed with the same tolerant comparison used for selection.
    component_path_ : PiPLSComponentPath
        Immutable concise view with one conditionally selected predictor-rank
        result per paired-mode count. Use :meth:`predictor_rank_profile` for the
        evaluated rank candidates at one paired-mode count.
    best_index_ : int
        Row of ``cv_results_`` selected by maximum mean test score, with smaller
        paired-mode count and predictor rank used as deterministic tie-breakers.
    best_score_ : float
        Mean cross-validation score at ``best_index_``.
    best_n_components_ : int
        Selected number of paired latent modes.
    best_predictor_rank_ : int
        Selected predictor rank.
    best_params_ : dict of str to int
        Parameters required to configure the supplied estimator or pipeline.
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
        n_jobs: int | None = None,
    ) -> None:
        self.estimator = estimator
        self.n_components_values = n_components_values
        self.predictor_rank_values = predictor_rank_values
        self.max_predictor_rank = max_predictor_rank
        self.search_method = search_method
        self.samples_per_predictor_rank = samples_per_predictor_rank
        self.cv = cv
        self.scoring = scoring
        self.n_jobs = n_jobs

    def fit(
        self,
        X: ArrayLike,
        y: ArrayLike,
        *,
        groups: ArrayLike | None = None,
    ) -> PiPLSSearchCV:
        """Evaluate the admissible Pi-PLS rank path.

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
        template = (
            _default_pipls_template() if self.estimator is None else self.estimator
        )
        pipls_param_prefix = _resolve_pipls_param_prefix(template)
        scorer = self._validate_constructor_parameters(template)
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

        n_components_key, predictor_rank_key = _pipls_parameter_keys(
            pipls_param_prefix
        )
        template = clone(template)
        materialized = _materialize_cv_splits(self.cv, X_array, y_array, groups=groups)
        self._cv_splits_ = _read_only_cv_splits(materialized.splits)
        self._n_samples_fit_ = int(X_array.shape[0])
        _validate_singleton_fold_scoring(self.scoring, self._cv_splits_)
        self.n_splits_ = len(self._cv_splits_)
        self.cv_n_train_min_ = materialized.n_train_min
        fold_feature_limit, fold_numerical_rank_limit = _fold_predictor_limits(
            template=template,
            X=X_indexable,
            y=y_indexable,
            splits=self._cv_splits_,
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
            if not any(candidate_h == int(h) for candidate_h, _ in admissible)
        ]
        if missing_components:
            missing = ", ".join(map(str, missing_components))
            raise ValueError(
                "Every n_components value must have at least one admissible predictor "
                f"rank; missing ranks for {missing}."
            )
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
                splits=self._cv_splits_,
                n_jobs=self.n_jobs,
            )
        else:
            for h_value in component_values:
                h = int(h_value)
                allowed = np.asarray(
                    [
                        candidate_rank
                        for candidate_h, candidate_rank in admissible
                        if candidate_h == h
                    ],
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
                    splits=self._cv_splits_,
                    n_jobs=self.n_jobs,
                )

        evaluated_pairs = tuple(sorted(cache))
        self.search_is_exhaustive_ = len(evaluated_pairs) == len(admissible)

        self.cv_results_ = _build_path_cv_results(
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

        self.component_path_ = _build_component_path(
            results=self.cv_results_,
            conditional_indices=np.asarray(conditional_indices, dtype=np.intp),
            predictor_rank_policy=predictor_rank_policy,
            n_splits=self.n_splits_,
        )
        return self

    def select(
        self,
        *,
        rule: SelectionRule | None = None,
        n_components: int | None = None,
    ) -> PiPLSComponentResult:
        """Return one immutable selected component-path result.

        Exactly one of ``rule`` and ``n_components`` must be supplied. A named
        rule selects one stored component-path row; a component count retrieves
        that row directly. In every case, the returned predictor rank is the
        rank already selected conditionally for the chosen component count.

        Parameters
        ----------
        rule : {"best_score", "minimum_cv_mse", "one_standard_error"}, optional
            Stored-row selection rule. ``"best_score"`` uses the global
            configured-score optimum, ``"minimum_cv_mse"`` uses the smallest
            stored mean response-standardized CV-MSE, and
            ``"one_standard_error"`` uses the smallest stored component count
            within one fold-based standard error of that minimum.
        n_components : int, optional
            Evaluated paired-mode count to retrieve manually.

        Returns
        -------
        PiPLSComponentResult
            Immutable stored result for the selected paired-mode count and its
            conditionally selected predictor rank.

        Raises
        ------
        sklearn.exceptions.NotFittedError
            If the search has not been fitted.
        ValueError
            If exactly one selection input is not supplied, a rule is invalid,
            or the requested component count was not evaluated.

        Notes
        -----
        The method performs no fitting or rescoring, does not mutate the search,
        and does not attach selected state.
        """

        return self._resolve_selection_result(
            rule=rule,
            n_components=n_components,
        )

    def refit(
        self,
        X: ArrayLike,
        y: ArrayLike,
        *,
        rule: SelectionRule | None = None,
        n_components: int | None = None,
    ) -> Any:
        """Fit and return one selected path model on the supplied full data.

        Exactly one of ``rule`` and ``n_components`` must be supplied. A named
        rule selects one stored component-path row; a component count retrieves
        that row directly. In every case, the predictor rank is the rank already
        selected conditionally for the chosen component count.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Predictor matrix used for the final full-data fit.
        y : array-like of shape (n_samples,) or (n_samples, n_targets)
            Response vector or matrix used for the final full-data fit.
        rule : {"best_score", "minimum_cv_mse", "one_standard_error"}, optional
            Stored-row selection rule. ``"best_score"`` uses the global
            configured-score optimum, ``"minimum_cv_mse"`` uses the smallest
            stored mean response-standardized CV-MSE, and
            ``"one_standard_error"`` uses the smallest stored component count
            within one fold-based standard error of that minimum.
        n_components : int, optional
            Evaluated paired-mode count to refit manually.

        Returns
        -------
        estimator
            Fitted clone of the configured direct estimator or pipeline. The
            returned outer estimator exposes the exact immutable selected row as
            ``selection_``.

        Raises
        ------
        sklearn.exceptions.NotFittedError
            If the search has not been fitted.
        ValueError
            If exactly one selection input is not supplied, a rule is invalid,
            or the requested component count was not evaluated.

        Notes
        -----
        The search object is not mutated and does not retain ``X``, ``y``, or
        the returned estimator. ``selection_`` is attached only after the
        full-data fit succeeds. A :class:`pipls.PiPLSRegression` fitted directly
        through :meth:`~pipls.PiPLSRegression.fit` has no selection provenance.
        For an exact manually specified ``(n_components, predictor_rank)`` pair,
        fit :class:`pipls.PiPLSRegression` directly.
        """

        selected = self._resolve_selection_result(
            rule=rule,
            n_components=n_components,
        )
        template = (
            _default_pipls_template() if self.estimator is None else self.estimator
        )
        pipls_param_prefix = _resolve_pipls_param_prefix(template)
        n_components_key, predictor_rank_key = _pipls_parameter_keys(
            pipls_param_prefix
        )
        estimator = clone(template).set_params(
            **{
                n_components_key: selected.n_components,
                predictor_rank_key: selected.predictor_rank,
            }
        )
        _fit_path_estimator(estimator, X, y)
        estimator.selection_ = selected
        return estimator

    def oof_report(
        self,
        X: ArrayLike,
        y: ArrayLike,
        *,
        selection: PiPLSComponentResult,
    ) -> PiPLSOOFReport:
        """Return ordered OOF diagnostics for one existing selection.

        The supplied selection is validated against this fitted search, then its
        fixed component-count and predictor-rank pair is fitted independently on
        every training fold from the exact split set materialized by :meth:`fit`.
        Repeated validation predictions are averaged, and rows never used for
        validation are represented by NaN with a zero prediction count.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Predictor matrix in the same row order and with the same sample
            count as the data supplied to :meth:`fit`.
        y : array-like of shape (n_samples,) or (n_samples, n_targets)
            Response vector or matrix aligned row-for-row with ``X`` and the
            data supplied to :meth:`fit`.
        selection : PiPLSComponentResult
            Existing immutable selection compatible with this fitted search.
            A model returned by :meth:`refit` exposes the exact value as
            ``model.selection_``.

        Returns
        -------
        PiPLSOOFReport
            Immutable report with the supplied selection, ordered OOF
            predictions, prediction counts, coverage provenance, and pooled OOF
            $R^2$ when at least two rows have validation coverage.

        Raises
        ------
        sklearn.exceptions.NotFittedError
            If the search has not been fitted.
        TypeError
            If ``selection`` is not a :class:`PiPLSComponentResult`.
        ValueError
            If the selection is incompatible with this search or the supplied
            data do not match the fitted search shape.

        Notes
        -----
        The report reuses the original validation indices but does not retain or
        compare the original data values. The caller is responsible for passing
        the same row-aligned observations. The search object is not mutated and
        no full-data model is fitted or retained.
        """

        compatible = self._validate_oof_selection(selection)
        values = self._compute_oof_report_values(
            X,
            y,
            selection=compatible,
            operation_name="oof_report",
        )
        return PiPLSOOFReport(
            selection=compatible,
            is_leave_one_out=values[0],
            oof_predictions=values[1],
            oof_prediction_counts=values[2],
            pooled_oof_r2=values[3],
        )

    def validation_report(
        self,
        X: ArrayLike,
        y: ArrayLike,
        *,
        rule: SelectionRule | None = None,
        n_components: int | None = None,
    ) -> PiPLSValidationReport:
        """Return the transitional report for one resolved path selection.

        New code should call :meth:`oof_report` with an existing selection. This
        method remains temporarily while maintained consumers migrate.
        """

        selected = self._resolve_selection_result(
            rule=rule,
            n_components=n_components,
        )
        values = self._compute_oof_report_values(
            X,
            y,
            selection=selected,
            operation_name="validation_report",
        )
        return PiPLSValidationReport(
            selected_result=selected,
            estimate_kind="selection-conditioned",
            is_leave_one_out=values[0],
            oof_predictions=values[1],
            oof_prediction_counts=values[2],
            pooled_oof_r2=values[3],
        )

    def _validate_oof_selection(
        self,
        selection: PiPLSComponentResult,
    ) -> PiPLSComponentResult:
        """Return a selection after exact compatibility validation."""

        if not isinstance(selection, PiPLSComponentResult):
            raise TypeError("selection must be a PiPLSComponentResult.")
        if selection.rule is None:
            expected = self._resolve_selection_result(
                rule=None,
                n_components=selection.n_components,
            )
        else:
            expected = self._resolve_selection_result(
                rule=selection.rule,
                n_components=None,
            )
        if selection != expected:
            raise ValueError(
                "selection is not compatible with this fitted search. Use "
                "model.selection_ or search.select(...) from the same search."
            )
        return selection

    def _compute_oof_report_values(
        self,
        X: ArrayLike,
        y: ArrayLike,
        *,
        selection: PiPLSComponentResult,
        operation_name: str,
    ) -> tuple[bool, FloatArray, IntArray, float | None]:
        """Compute common ordered OOF report values without mutation."""

        check_is_fitted(self, attributes=["_cv_splits_", "_n_samples_fit_"])
        validated = _validate_estimator_data(
            self,
            X,
            y,
            reset=False,
            accept_sparse=False,
            dtype=np.float64,
            multi_output=True,
            y_numeric=True,
            ensure_min_samples=2,
            copy=False,
        )
        X_checked, y_checked = cast(tuple[Any, Any], validated)
        y_array = np.asarray(y_checked, dtype=np.float64)
        if int(np.shape(X_checked)[0]) != self._n_samples_fit_:
            raise ValueError(
                f"{operation_name}() requires the same number of samples used "
                f"during search.fit(); expected {self._n_samples_fit_}, got "
                f"{np.shape(X_checked)[0]}."
            )
        n_targets = 1 if y_array.ndim == 1 else int(y_array.shape[1])
        if n_targets != self.n_targets_:
            raise ValueError(
                f"{operation_name}() requires the same number of response columns "
                f"used during search.fit(); expected {self.n_targets_}, got {n_targets}."
            )

        X_indexable, y_indexable = indexable(X, y)
        template = (
            _default_pipls_template() if self.estimator is None else self.estimator
        )
        pipls_param_prefix = _resolve_pipls_param_prefix(template)
        n_components_key, predictor_rank_key = _pipls_parameter_keys(
            pipls_param_prefix
        )
        oof_predictions, counts = _ordered_oof_predictions(
            candidate=_PiPLSCandidate(
                n_components=selection.n_components,
                predictor_rank=selection.predictor_rank,
            ),
            template=template,
            n_components_key=n_components_key,
            predictor_rank_key=predictor_rank_key,
            X=X_indexable,
            y=y_indexable,
            splits=self._cv_splits_,
            n_jobs=self.n_jobs,
            ignored_warning_categories=_CONTROLLED_FIT_WARNING_CATEGORIES,
        )
        predictions = oof_predictions[:, 0] if y_array.ndim == 1 else oof_predictions
        return (
            _splits_are_leave_one_out(
                self._cv_splits_,
                n_samples=self._n_samples_fit_,
            ),
            predictions,
            counts,
            _pooled_oof_r2(y_indexable, predictions, counts),
        )

    def _resolve_selection_result(
        self,
        *,
        rule: SelectionRule | None,
        n_components: int | None,
    ) -> PiPLSComponentResult:
        """Resolve one stored component-path row without mutating search state."""

        check_is_fitted(
            self,
            attributes=["cv_results_", "component_path_", "best_n_components_"],
        )
        if (rule is None) == (n_components is None):
            raise ValueError(
                "Exactly one of rule and n_components must be supplied."
            )
        if n_components is not None:
            return _component_result_at_count(
                self.component_path_,
                n_components,
            )
        if rule == "best_score":
            return replace(
                _component_result_at_count(
                    self.component_path_,
                    self.best_n_components_,
                ),
                rule="best_score",
            )
        if rule == "minimum_cv_mse":
            return _select_minimum_cv_mse(self.component_path_)
        if rule == "one_standard_error":
            return _select_one_standard_error(self.component_path_)
        raise ValueError(
            'rule must be "best_score", "minimum_cv_mse", or '
            '"one_standard_error".'
        )

    def predictor_rank_profile(
        self,
        n_components: int,
    ) -> PiPLSPredictorRankProfile:
        """Return evaluated predictor-rank results for one paired-mode count.

        Rows are sorted by ascending predictor rank and include only candidates
        actually evaluated by the fitted search. The selected scalar result
        maximizes the configured mean test score, with the fitted conditional
        tie-breaking rule. Under the default scorer, this is equivalent to
        minimizing mean response-standardized CV-MSE.

        Parameters
        ----------
        n_components : int
            Evaluated paired-mode count whose predictor-rank profile is requested.

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
        selected = _component_result_at_count(
            self.component_path_,
            n_components,
        )
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
            predictor_rank_policy=selected.predictor_rank_policy,
            n_splits=self.n_splits_,
        )

    def _more_tags(self) -> dict[str, bool]:
        """Legacy scikit-learn tags for releases before the Tags dataclasses."""

        return {"multioutput": True}

    def __sklearn_tags__(self) -> Any:
        """Declare multi-output target support for the path evaluator."""

        parent = getattr(super(), "__sklearn_tags__", None)
        if parent is None:  # pragma: no cover - scikit-learn 1.4/1.5
            return self._more_tags()
        tags = parent()
        tags.target_tags.multi_output = True
        tags.target_tags.single_output = True
        return tags

    def _validate_constructor_parameters(self, template: Any) -> Scorer:
        if self.search_method not in ("optimal", "auto"):
            raise ValueError('search_method must be "optimal" or "auto".')
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
                PredictorRankSupportWarning,
                stacklevel=3,
            )
        if self.max_predictor_rank != "rule":
            _validate_positive_int(self.max_predictor_rank, name="max_predictor_rank")
        _validate_component_values(self.n_components_values)
        _validate_predictor_rank_values(self.predictor_rank_values)
        _validate_n_jobs(self.n_jobs)
        _validate_cv(self.cv)
        return _resolve_path_scorer(self.scoring, template)


def _read_only_cv_splits(splits: tuple[CVSplit, ...]) -> tuple[CVSplit, ...]:
    """Return defensive read-only copies of materialized split indices."""

    stored: list[CVSplit] = []
    for train, validation in splits:
        train_copy = np.array(train, dtype=np.intp, copy=True)
        validation_copy = np.array(validation, dtype=np.intp, copy=True)
        train_copy.flags.writeable = False
        validation_copy.flags.writeable = False
        stored.append((train_copy, validation_copy))
    return tuple(stored)


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
            final_estimator, X_transformed = _prepare_fold_pipls_inputs(
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
                _fit_path_estimator(probe, X_transformed, y_train)
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


def _prepare_fold_pipls_inputs(
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


def _fit_path_estimator(estimator: Any, X: ArrayLike, y: ArrayLike) -> None:
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


def _build_path_cv_results(
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


def _build_component_path(
    *,
    results: dict[str, Any],
    conditional_indices: IntArray,
    predictor_rank_policy: PredictorRankPolicy,
    n_splits: int,
) -> PiPLSComponentPath:
    """Return one conditionally selected predictor-rank result per paired-mode count."""

    return PiPLSComponentPath(
        n_components=cast(IntArray, results["n_components"])[conditional_indices],
        predictor_rank=cast(IntArray, results["predictor_rank"])[conditional_indices],
        predictor_rank_policy=predictor_rank_policy,
        mean_test_score=cast(FloatArray, results["mean_test_score"])[conditional_indices],
        cv_mse_mean=cast(
            FloatArray,
            results["mean_response_standardized_mse"],
        )[conditional_indices],
        cv_mse_fold_sd=cast(
            FloatArray,
            results["std_response_standardized_mse"],
        )[conditional_indices],
        n_splits=n_splits,
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
        return neg_response_standardized_mse
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
