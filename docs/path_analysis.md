# Path-selection details

The [synthetic tutorial](tutorials/synthetic.md#retrieve-selection-evidence) owns the routine
selection workflow. This page records configuration, cross-validation, and edge behavior needed
when the defaults are not enough. Exact signatures and fitted attributes are in the
[generated path API](api/path.md).

## Search bounds

For paired-mode count $h$ (`n_components`) and retained predictor-subspace dimension $r_\pi$
(`predictor_rank`), the admissible pairs are

\begin{equation}
\mathcal{G}=\{(h,r_\pi):1\le h\le h_{\mathrm{max}},\ h\le r_\pi\le r_{\pi,\mathrm{max}}\}.
\end{equation}

### Resolved ceilings

The default upper predictor rank is

\begin{equation}
r_{\pi,\mathrm{max}}=\min\left[p_{\mathrm{min}},n_{\mathrm{train,min}}-1,
r_{\mathrm{num,min}},
\left\lceil\frac{n}{c}\right\rceil\right].
\end{equation}

Here $n$ is the total number of observations supplied to `fit()`, $p_{\mathrm{min}}$ is the minimum
predictor count after fold-local pipeline preprocessing, $r_{\mathrm{num,min}}$ is the minimum
verified predictor rank after terminal-estimator centering and optional scaling, and $c$ is `samples_per_predictor_rank`. The search object fits
pipeline preprocessing separately inside each fold before this rank preflight. An integer
`max_predictor_rank` bypasses the statistical support rule but remains capped by fold dimensions and
numerical rank.

If the response matrix has $q$ columns, the resolved component ceiling is

\begin{equation}
h_{\mathrm{max}}=\min(q,r_{\pi,\mathrm{max}}).
\end{equation}

This ceiling enforces $h\le q$ and guarantees that at least one predictor rank can satisfy
$h\le r_\pi$ before an explicit predictor-rank set is applied. Explicit component or predictor-rank
values above the corresponding resolved ceiling are rejected before candidate evaluation. If an
explicit predictor-rank set leaves a requested component count without any rank satisfying
$h\le r_\pi$, the request is rejected rather than silently dropping that component count.

### Component-count requests

The default `n_components_values="all"` evaluates every paired-mode count from 1 through
$h_{\mathrm{max}}$. An explicit integer sequence requests a subset:

```python
search = PiPLSSearchCV(
    n_components_values=[1, 2, 3, 5],
).fit(X, Y)
```

Every requested value must have at least one admissible predictor rank.

## Predictor-rank policies

With `predictor_rank_values=None`, predictor rank is selected independently for every component
count. A one-element sequence fixes one rank across the path, a longer sequence defines the
admissible set, and `predictor_rank_values="max"` uses $r_{\pi,\mathrm{max}}$ directly.

`search_method="optimal"` evaluates every admissible pair. `search_method="auto"` performs a
deterministic adaptive coarse-to-fine search independently for each component count and may leave
admissible ranks unevaluated. After fitting, `search_is_exhaustive_` states whether every
admissible pair was evaluated. Score ties within numerical tolerance favor the smaller predictor
rank for a fixed component count. The global best then favors the smaller component count and the
smaller predictor rank.

## Scoring and the best evaluated pair

Candidate selection maximizes the configured mean test score. The default scoring parameter is
the stable package name `"neg_response_standardized_mse"`, which resolves to
`pipls.metrics.neg_response_standardized_mse`. Maximizing that score is equivalent
to minimizing mean response-standardized CV-MSE. With another scorer, the CV-MSE columns remain
diagnostics and need not identify the selected candidate.

Ordinary scikit-learn scorer names, scorer callables, and `scoring=None` are accepted. Retrieve
the best evaluated pair through `best = search.select(rule="best_score")`; its component count,
predictor rank, and mean score are available on that immutable selection. `rank_test_score` uses
minimum ranks with the same `rtol=1e-12` and `atol=1e-15` comparison as
selection. Every rank-1 candidate is tied directly with the maximum score; lower rank groups are
likewise anchored to their leading score rather than formed through adjacent-score chaining.
Adaptive search makes no claim about pairs it did not evaluate. The component path remains a
model-selection diagnostic; its numerical minimum does not replace a scientifically justified
complexity choice.

## Pipelines and fold-local preprocessing

The supported estimator is either a direct `PiPLSRegression` or a scikit-learn `Pipeline` whose
final step is `PiPLSRegression`. The complete estimator is cloned and fitted inside every fold.
For the direct-template pattern and inherited settings, see
[Configure the candidate estimator](api/path.md#configure-the-candidate-estimator).

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
before fold-rank preflight, every candidate fit, explicit OOF reporting, and each post-fit refit,
so the seed pair does not restrict or select the path. Other template settings, including `scale`,
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

Request ordered OOF diagnostics for an existing selection after modeling or selection-only analysis:

```python
from sklearn.model_selection import KFold

cv = KFold(n_splits=5, shuffle=True, random_state=0)
search = PiPLSSearchCV(cv=cv).fit(X, Y)
model = search.refit(X, Y, rule="minimum_cv_mse")

selection = model.selection_
report = search.oof_report(X, Y, selection=selection)
```

A workflow that deliberately fits no final model may obtain the selection directly:

```python
selection = search.select(n_components=4)
report = search.oof_report(X, Y, selection=selection)
```

The method validates that the supplied selection belongs to the fitted search, fits that fixed
parameterization once per stored training fold, and returns an immutable report:

- `selection` is the exact supplied component-count and predictor-rank selection;
- `oof_predictions` preserves input row order;
- repeated validation predictions are averaged and their counts are recorded;
- rows without validation coverage have count 0 and a NaN prediction;
- selection metrics remain available through `report.selection`;
- `pooled_oof_r2` uses only rows with OOF coverage.

`fit()` stores defensive read-only copies of the exact materialized validation indices. Therefore an
iterable splitter is not consumed a second time and a stochastic splitter is not asked to generate a
new partition. `oof_report()` requires the same sample count, feature count, and response-column
count as the fitted search, but it does not retain or compare original values. The caller is
responsible for passing the same observations in the same row order.

The report separately records OOF coverage and whether the splitter is structurally leave-one-out.
These results are selection-conditioned because the same path search produced the supplied
selection. Use nested cross-validation or an external test set when an unbiased post-selection
estimate is required.

## Leave-one-out interpretation

`LeaveOneOut()` is not a special Pi-PLS mode. The support term uses the full $n$, while centered-fold
feasibility is capped by $n-2$. The default standardized MSE remains defined for singleton
validation folds because response scales are estimated from each training fold.

Mean foldwise $R^2$ is rejected when validation folds contain one sample. The explicit report's
`pooled_oof_r2` may report $R^2$ from pooled LOO predictions; it is not mean foldwise $R^2$.

The [focused small-sample example](examples.md#leave-one-out-validation) uses twelve deterministic
observations, a compact explicit candidate grid, the singleton-safe default scorer, and ordered OOF
predictions. Its OOF report is selection-conditioned because the same LOO path selects the rank
pair and supplies the pooled diagnostic.

## Split variation and tolerance selection

`component_path_.cv_mse_std` is the population standard deviation of the realized split-specific
MSE values. It describes variation across the materialized validation splits. Maintained
component-path and predictor-rank-profile figures plot mean response-standardized CV-MSE with
symmetric $\pm 1$ SD bars from this stored quantity. The bars are descriptive split-to-split
variability; they are not confidence intervals and do not enter selection. The path and profile
objects expose no standard-error property.

### Search-owned selection rules

`PiPLSSearchCV.select()` returns complete immutable stored rows without fitting:

```python
minimum = search.select(rule="minimum_cv_mse")
tolerant = search.select(
    rule="minimum_cv_mse",
    relative_tolerance=0.02,
    absolute_tolerance=np.inf,
)
best = search.select(rule="best_score")
manual = search.select(n_components=3)
```

For the minimum-CV-MSE rule, let $M_{\min}$ be the exact minimum stored mean CV-MSE. A row qualifies
only when its mean is no larger than both $(1+\delta_{\mathrm{rel}})M_{\min}$ and
$M_{\min}+\delta_{\mathrm{abs}}$. The first qualifying row is returned because component counts
are stored in strictly ascending order. `relative_tolerance=None` resolves to
`sqrt(np.finfo(np.float64).eps)`, while positive-infinity `absolute_tolerance` disables the absolute
cap. The result retains the exact unruled minimum row as `reference_minimum`, the resolved
tolerances, and the derived `cv_mse_threshold`. Direct lookup by component count and `best_score`
selection carry no tolerance provenance.

The associated predictor rank is the rank already selected conditionally for that component count
under the configured scorer. `select()` does not revisit the predictor-rank profile, fit or refit an
estimator, mutate the search object, or attach selected state. With a nondefault scorer, the stored
predictor rank need not minimize CV-MSE within its component-count profile. Model-producing
workflows obtain the fitted row from `model.selection_`; `search.select(...)` remains useful for
selection-only analysis. `component_path_` remains the aligned numerical curve.

The Tobacco workflow applies `rule="minimum_cv_mse"` with `relative_tolerance=0.10` once through
`refit()`. The returned `model.selection_` carries the selected row, exact `reference_minimum`,
resolved tolerance, and derived `cv_mse_threshold`. The same selection supplies the conditional
predictor-rank profile and `oof_report()`. Predictor-rank profile error bars use the stored split SD,
while the stored predictor rank for each component count continues to maximize the configured mean
CV score. The optional `absolute_tolerance` remains at positive infinity in this example.

## Post-fit final-model selection

After path evaluation, `refit()` selects one stored component-path row and fits the corresponding
fixed model on the supplied full data:

```python
search = PiPLSSearchCV(search_method="auto").fit(X, Y)
model = search.refit(X, Y, rule="minimum_cv_mse")
Y_pred = model.predict(X_new)
```

Manual selection uses the same operation:

```python
model = search.refit(X, Y, n_components=4)
```

Exactly one of `rule` and `n_components` is required. The accepted post-fit rules are:

- `rule="best_score"`, the global optimum under the configured scorer;
- `rule="minimum_cv_mse"`, the smallest stored component-path row satisfying simultaneous relative
  and absolute tolerances around the exact minimum mean response-standardized CV-MSE.

Each rule retains the predictor rank already selected conditionally for the chosen component count.
With a nondefault scorer, that rank remains conditioned on the scorer even when the component rule
uses response-standardized CV-MSE. Relative tolerance must be finite and nonnegative; absolute
tolerance must be nonnegative and may be positive infinity. Nondefault tolerance arguments apply
only to `rule="minimum_cv_mse"`.

`refit()` clones the configured direct estimator or pipeline, replaces the terminal Pi-PLS rank
pair, fits the clone, attaches the exact immutable row as `model.selection_`, and returns the model.
It does not mutate the search, store the supplied matrices, or attach the model to search state.
Prediction, transformation, scoring, inverse transformation, and feature-name behavior belong to
the returned model.

`oof_report()` consumes an existing selection and does not alter search state. The search stores
candidate evidence and reusable split indices, but no report or fitted final model.

## Search and model diagnostics

Use `component_path_` for the concise component-count curve and `predictor_rank_profile(h)` for the
evaluated ranks at one count. `cv_results_` contains aligned `n_components` and `predictor_rank`
arrays, split and summary test scores, response-standardized MSE diagnostics, score ranks, and
fit/score timing summaries. The direct parameter columns are stable for both direct estimators and
pipelines, and the dictionary contains only evaluated candidates.

Use the returned estimator for fixed-model diagnostics such as `decomposition_`, coefficients,
latent scores, predictions, and inspection helpers. Retain the search variable when both search
evidence and the final fitted model are needed.
