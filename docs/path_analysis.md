# Path-selection details

The [synthetic tutorial](tutorials/synthetic.md#evaluate-the-component-path) owns the routine
selection workflow. This page records configuration, cross-validation, and edge behavior needed
when the defaults are not enough. Exact signatures and fitted attributes are in the
[generated path API](api/path.md).

## Search bounds

For component count $h$ and predictor rank $r_\pi$, the admissible pairs are

\begin{equation}
\mathcal{G}=\{(h,r_\pi):1\le h\le h_{\max},\ h\le r_\pi\le r_{\pi,\max}\}.
\end{equation}

### Resolved ceilings

The default upper predictor rank is

\begin{equation}
r_{\pi,\max}=\min\left[p_{\min},n_{\mathrm{train,min}}-1,
r_{\mathrm{num,min}},
\left\lceil\frac{n}{\texttt{samples_per_predictor_rank}}\right\rceil\right].
\end{equation}

Here $n$ is the total number of observations supplied to `fit()`, $p_{\min}$ is the minimum
predictor count after fold-local pipeline preprocessing, and $r_{\mathrm{num,min}}$ is the minimum
verified predictor rank after terminal-estimator centering and optional scaling. The search object fits
pipeline preprocessing separately inside each fold before this rank preflight. An integer
`max_predictor_rank` bypasses the statistical support rule but remains capped by fold dimensions and
numerical rank.

If the response matrix has $q$ columns, the resolved component ceiling is

\begin{equation}
h_{\max}=\min(q,r_{\pi,\max}).
\end{equation}

This ceiling enforces $h\le q$ and guarantees that at least one predictor rank can satisfy
$h\le r_\pi$ before an explicit predictor-rank set is applied. Explicit component or predictor-rank
values above the corresponding resolved ceiling are rejected before candidate evaluation. If an
explicit predictor-rank set leaves a requested component count without any rank satisfying
$h\le r_\pi$, the request is rejected rather than silently dropping that component count.

### Component-count requests

The default `n_components_values="all"` evaluates every component count from 1 through
$h_{\max}$. An explicit integer sequence requests a subset:

```python
search = PiPLSSearchCV(
    n_components_values=[1, 2, 3, 5],
).fit(X, Y)
```

Every requested value must have at least one admissible predictor rank.

## Predictor-rank policies

With `predictor_rank_values=None`, predictor rank is selected independently for every component
count. A one-element sequence fixes one rank across the path, a longer sequence defines the
admissible set, and `predictor_rank_values="max"` uses $r_{\pi,\max}$ directly.

`search_method="optimal"` evaluates every admissible pair. `search_method="auto"` performs a
deterministic adaptive coarse-to-fine search independently for each component count and may leave
admissible ranks unevaluated. After fitting, `path_search_exhaustive_` states whether every
admissible pair was evaluated. Score ties within numerical tolerance favor the smaller predictor
rank for a fixed component count. The global best then favors the smaller component count and the
smaller predictor rank.

## Scoring and the best evaluated pair

Candidate selection maximizes the configured mean test score. The default scoring parameter is
the stable package name `"neg_response_standardized_mean_squared_error"`, which resolves to
`pipls.metrics.neg_response_standardized_mean_squared_error`. Maximizing that score is equivalent
to minimizing mean response-standardized CV-MSE. With another scorer, the CV-MSE columns remain
diagnostics and need not identify the selected candidate.

Ordinary scikit-learn scorer names, scorer callables, and `scoring=None` are accepted.
`best_params_`, `best_score_`, and `best_index_` describe the best evaluated pair.
`rank_test_score` uses minimum ranks with the same `rtol=1e-12` and `atol=1e-15` comparison as
selection. Every rank-1 candidate is tied directly with the maximum score; lower rank groups are
likewise anchored to their leading score rather than formed through adjacent-score chaining.
Adaptive search makes no claim about pairs it did not evaluate. The component path remains a
model-selection diagnostic; its numerical minimum does not replace a scientifically justified
complexity choice.

## Pipelines and fold-local preprocessing

The supported estimator is either a direct `PiPLSRegression` or a scikit-learn `Pipeline` whose
final step is `PiPLSRegression`. The complete estimator is cloned and fitted inside every fold.

```python
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from pipls import PiPLSRegression, PiPLSSearchCV

pipeline = Pipeline(
    [
        ("impute", SimpleImputer()),
        ("pipls", PiPLSRegression(n_components=1, predictor_rank=1)),
    ]
)
search = PiPLSSearchCV(estimator=pipeline).fit(X, Y)
```

The terminal estimator needs the valid construction seed pair `(1, 1)` because
`PiPLSRegression` always represents one explicit fixed pair. `PiPLSSearchCV` replaces both values
before fold-rank preflight, every candidate fit, optional OOF fitting, and the final refit, so the
seed pair does not restrict or select the path. Other template settings, including `scale`,
`svd_solver`, and `random_state`, do affect candidate fitting.

Do not fit learned preprocessing on the complete dataset before path evaluation.

## Cross-validation protocols and metadata

`PiPLSSearchCV` accepts scikit-learn splitters for grouped, repeated, predefined, temporal, or
leave-one-out protocols when their scientific assumptions match the data. Suitable examples include
`GroupKFold`, `RepeatedKFold`, `PredefinedSplit`, `TimeSeriesSplit`, and `LeaveOneOut`.

```python
from sklearn.model_selection import GroupKFold

search = PiPLSSearchCV(cv=GroupKFold(n_splits=5))
search.fit(X, Y, groups=sample_groups)
```

`groups` is an explicit `PiPLSSearchCV.fit()` parameter and participates in scikit-learn metadata
routing when routing is enabled and requested. Split metadata belongs to the search object;
`PiPLSRegression.fit(X, Y)` fits one explicit pair and accepts none.

## Ordered out-of-fold predictions

Set `return_oof_predictions=True` to fit `selected_result_` once per training fold after selection.
The immutable `validation_report_` then owns row-ordered OOF results:

- `oof_predictions` preserves input row order;
- repeated validation predictions are averaged and their counts are recorded;
- rows without validation coverage have count 0 and a NaN prediction;
- `n_components` and `predictor_rank` identify the fitted pair;
- `pooled_oof_r2` uses only rows with OOF coverage.

The report also records the split count, mean score, CV-MSE, OOF coverage, and whether the splitter
is structurally leave-one-out. These results are selection-conditioned because the same path search
selected the parameters. Use nested cross-validation or an external test set when an unbiased
post-selection estimate is required.

## Leave-one-out interpretation

`LeaveOneOut()` is not a special Pi-PLS mode. The support term uses the full $n$, while centered-fold
feasibility is capped by $n-2$. The default standardized MSE remains defined for singleton
validation folds because response scales are estimated from each training fold.

Mean foldwise $R^2$ is rejected when validation folds contain one sample. When OOF predictions are
requested, `validation_report_.pooled_oof_r2` may report $R^2$ from pooled LOO predictions; it is
not mean foldwise $R^2$.

The [focused small-sample example](examples.md#leave-one-out-validation) uses twelve deterministic
observations, a compact explicit candidate grid, the singleton-safe default scorer, and ordered OOF
predictions. Its validation report is selection-conditioned because the same LOO path selects the
rank pair and supplies the pooled diagnostic.

## Fold variation and standard error

`component_path_.cv_mse_fold_sd` is the population standard deviation of the realized fold-specific
MSE values. It describes fold variation. The derived read-only
`component_path_.cv_mse_standard_error` converts that stored quantity to the usual fold-based
standard error of the mean CV-MSE:

\[
\widehat{\mathrm{SE}}_{\mathrm{CV}}
=
\frac{\widehat{\sigma}_{\mathrm{fold,pop}}}{\sqrt{K-1}},
\]

where $K$ is the number of validation splits. This is equivalent to dividing the sample standard
deviation of the fold MSE values by $\sqrt{K}$. At least two splits are required. Maintained
component-path and predictor-rank-profile figures plot the mean response-standardized CV-MSE with
symmetric $\pm 1$ standard-error bars from this property. Because CV training sets overlap, these
bars are a conventional resampling heuristic rather than confidence intervals or a formal
uncertainty guarantee.

### One-standard-error component heuristic

The conventional one-standard-error rule, usually abbreviated the 1-SE rule, can use the component
path to favor a more parsimonious component count. Let $h_{\min}$ minimize the displayed mean
CV-MSE and define

\[
\tau
=
\widehat{\operatorname{CV\text{-}MSE}}(h_{\min})
+
\widehat{\operatorname{SE}}_{\mathrm{CV}}(h_{\min}).
\]

The rule chooses the smallest evaluated component count whose mean CV-MSE does not exceed $\tau$.
It is a heuristic for identifying a simpler model within one estimated standard error of the
minimum; it does not establish equivalence between the candidates. It is particularly convenient
when the CV-MSE curve has no clear elbow that would otherwise motivate a component count. The
[Tobacco one-standard-error workflow](examples.md#tobacco-one-standard-error-selection) demonstrates this case and
shows the minimum row, horizontal threshold, and recommended row in the component-path figure.

### Result-object recommendations

`PiPLSComponentPath` provides two non-mutating reference methods for these stored path choices:

```python
minimum = search.component_path_.minimum_cv_mse_result()
one_se = search.component_path_.one_standard_error_result()
```

Both methods return a complete immutable `PiPLSComponentResult`. `minimum_cv_mse_result()` returns
the first stored row attaining the exact minimum mean CV-MSE. Because component counts are stored in
strictly ascending order, an exact tie returns the smallest tied count.
`one_standard_error_result()` returns the first stored row satisfying the 1-SE threshold above.
It requires at least two validation splits so that the reference-row standard error is defined.

The associated predictor rank is the rank already selected conditionally for that component count
under the configured scorer. The methods do not revisit the predictor-rank profile, fit or refit an
estimator, mutate the search object, or alter `best_*`. With a nondefault scorer, the stored
predictor rank need not minimize CV-MSE within its component-count profile. The returned rows can be
used directly for user judgment or by the explicit path-level selection rule described below.

The Tobacco workflow calls `one_standard_error_result()` explicitly and uses the returned component
count and the predictor rank already stored in that component-path row to fit the final fixed model.
The other maintained examples and tutorials retain explicit component choices. Conditional
predictor-rank profiles use the same standard-error bars for scale, but the stored predictor rank
for each component count continues to maximize the configured mean CV score rather than applying
the 1-SE rule.

## Predeclared final-model selection

When the model-building protocol is known before fitting, `selection_rule` can choose the final
stored component-path row without introducing selection into `PiPLSRegression`:

```python
search = PiPLSSearchCV(
    search_method="auto",
    selection_rule="one_standard_error",
    refit=True,
).fit(X, Y)

model = search.selected_pipls_
Y_pred = model.predict(X_new)
```

Once the selected Pi-PLS model has been extracted, application code should call its methods
directly. Delegated methods such as `search.predict(X_new)` remain available as scikit-learn-style
conveniences, but the explicit model call keeps selection and model application visually separate.

The accepted rules are:

- `selection_rule="best_score"`, the default, which chooses the globally best evaluated pair under
  the configured scorer;
- `selection_rule="one_standard_error"`, which chooses the exact stored row returned by
  `component_path_.one_standard_error_result()`.

The second rule therefore selects $h$ from the stored CV-MSE path and retains the predictor rank
already selected conditionally for that $h$. With a nondefault scorer, the rank remains conditioned
on that scorer even though the component rule uses response-standardized CV-MSE. The rule requires
at least two validation splits.

The global optimum and the declared final choice remain separate results. `best_index_`,
`best_score_`, `best_params_`, `best_n_components_`, and `best_predictor_rank_` always describe the
global configured-score optimum. `selected_result_` and `selected_params_` describe the final path
row. `validation_report_` and optional OOF predictions also represent that selected row.

This one-call form is suitable only for a rule declared in advance. A component count chosen after
examining the path is a post-hoc scientific decision and should remain an explicit second fit.

## Refit and detailed diagnostics

The default `refit=False` leaves path evaluation and final fixed-model fitting as separate steps,
which keeps the component-count choice visible. With `refit=True`, the row chosen by
`selection_rule` is fitted on all supplied data as `selected_estimator_` and `selected_pipls_`, and
supported prediction or transformation methods delegate to it. Under the default
`selection_rule="best_score"`, `best_estimator_` and `best_pipls_` remain compatibility aliases. They
are absent for the 1-SE rule because that refitted model is a declared recommendation rather than
the global score optimum.

Use `component_path_` for the concise component-count curve and `predictor_rank_profile(h)` for the
evaluated ranks at one count. `cv_results_` contains candidate parameters, split test scores,
response-standardized MSE diagnostics, score ranks, and fit/score timing summaries. It contains only
evaluated candidates.
