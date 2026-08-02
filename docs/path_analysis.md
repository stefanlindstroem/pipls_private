# Path-selection details

The [synthetic tutorial](tutorials/synthetic.md#evaluate-the-component-path) owns the routine
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

Request ordered OOF diagnostics explicitly after path evaluation:

```python
from sklearn.model_selection import KFold

cv = KFold(n_splits=5, shuffle=True, random_state=0)
search = PiPLSSearchCV(cv=cv).fit(X, Y)
report = search.validation_report(X, Y, rule="one_standard_error")
# or: report = search.validation_report(X, Y, n_components=4)
```

The method uses the same selected-row resolver as `refit()` but performs no full-data fit. It fits the
selected fixed parameterization once per stored training fold and returns an immutable report:

- `selected_result` is the complete stored component-path row selected by the requested rule or count;
- `oof_predictions` preserves input row order;
- repeated validation predictions are averaged and their counts are recorded;
- rows without validation coverage have count 0 and a NaN prediction;
- `n_components` and `predictor_rank` remain convenient views of the selected result;
- `pooled_oof_r2` uses only rows with OOF coverage.

`fit()` stores defensive read-only copies of the exact materialized validation indices. Therefore an
iterable splitter is not consumed a second time and a stochastic splitter is not asked to generate a
new partition. `validation_report()` requires the same sample count, feature count, and response-column
count as the fitted search, but it does not retain or compare original values. The caller is responsible
for passing the same observations in the same row order.

The report derives its split count, mean score, and CV-MSE convenience attributes from
`selected_result` and separately records OOF coverage and whether the splitter is structurally
leave-one-out. These results are selection-conditioned because the same path search selected the
parameters. Use nested cross-validation or an external test set when an unbiased post-selection
estimate is required.

## Leave-one-out interpretation

`LeaveOneOut()` is not a special Pi-PLS mode. The support term uses the full $n$, while centered-fold
feasibility is capped by $n-2$. The default standardized MSE remains defined for singleton
validation folds because response scales are estimated from each training fold.

Mean foldwise $R^2$ is rejected when validation folds contain one sample. The explicit report's
`pooled_oof_r2` may report $R^2$ from pooled LOO predictions; it is not mean foldwise $R^2$.

The [focused small-sample example](examples.md#leave-one-out-validation) uses twelve deterministic
observations, a compact explicit candidate grid, the singleton-safe default scorer, and ordered OOF
predictions. Its validation report is selection-conditioned because the same LOO path selects the
rank pair and supplies the pooled diagnostic.

## Fold variation and standard error

`component_path_.cv_mse_fold_sd` is the population standard deviation of the realized fold-specific
MSE values. It describes fold variation. The derived read-only
`component_path_.cv_mse_standard_error` converts that stored quantity to the usual fold-based
standard error of the mean CV-MSE:

\begin{equation}
\widehat{\mathrm{SE}}_{\mathrm{CV}}
=
\frac{\widehat{\sigma}_{\mathrm{fold,pop}}}{\sqrt{K-1}},
\end{equation}

where $K$ is the number of validation splits, stored once as the path-wide scalar
`component_path_.n_splits`. This is equivalent to dividing the sample standard deviation of the
fold MSE values by $\sqrt{K}$. At least two splits are required. Maintained
component-path and predictor-rank-profile figures plot the mean response-standardized CV-MSE with
symmetric $\pm 1$ standard-error bars from this property. Because CV training sets overlap, these
bars are a conventional resampling heuristic rather than confidence intervals or a formal
uncertainty guarantee.

### One-standard-error component heuristic

The conventional one-standard-error rule, usually abbreviated the 1-SE rule, can use the component
path to favor a more parsimonious component count. Let $h_{\mathrm{min}}$ minimize the displayed mean
CV-MSE and define

\begin{equation}
\tau
=
\widehat{\operatorname{CV\text{-}MSE}}(h_{\mathrm{min}})
+
\widehat{\operatorname{SE}}_{\mathrm{CV}}(h_{\mathrm{min}}).
\end{equation}

The rule chooses the smallest evaluated component count whose mean CV-MSE does not exceed $\tau$.
It is a heuristic for identifying a simpler model within one estimated standard error of the
minimum; it does not establish equivalence between the candidates. It is particularly convenient
when the CV-MSE curve has no clear elbow that would otherwise motivate a component count. The
[Tobacco one-standard-error workflow](examples.md#tobacco-one-standard-error-selection) demonstrates this case and
shows the minimum row, horizontal threshold, and recommended row in the component-path figure.

### Search-owned selection rules

`PiPLSSearchCV.select()` returns complete immutable stored rows for these choices without fitting:

```python
minimum = search.select(rule="minimum_cv_mse")
one_se = search.select(rule="one_standard_error")
manual = search.select(n_components=3)
```

The minimum rule returns the first stored row attaining the exact minimum mean CV-MSE. Because
component counts are stored in strictly ascending order, an exact tie returns the smallest tied
count. The 1-SE rule returns the first stored row satisfying the threshold above and requires at
least two validation splits so that the reference-row standard error is defined.

The associated predictor rank is the rank already selected conditionally for that component count
under the configured scorer. `select()` does not revisit the predictor-rank profile, fit or refit an
estimator, mutate the search object, or alter `best_*`. With a nondefault scorer, the stored
predictor rank need not minimize CV-MSE within its component-count profile. Maintained workflows
use `search.select(...)` for scalar annotations and reporting, while `component_path_` remains the
aligned numerical curve.

The Tobacco workflow resolves both the minimum row and the one-standard-error recommendation
through `search.select(...)`, then applies the same named 1-SE rule independently through both
`refit()` and `validation_report()`. Conditional predictor-rank profiles use the same standard-error
bars for scale, but the stored predictor rank for each component count continues to maximize the
configured mean CV score rather than applying the 1-SE rule.

## Post-fit final-model selection

After path evaluation, `refit()` selects one stored component-path row and fits the corresponding
fixed model on the supplied full data:

```python
search = PiPLSSearchCV(search_method="auto").fit(X, Y)
model = search.refit(X, Y, rule="one_standard_error")
Y_pred = model.predict(X_new)
```

Manual selection uses the same operation:

```python
model = search.refit(X, Y, n_components=4)
```

Exactly one of `rule` and `n_components` is required. The accepted post-fit rules are:

- `rule="best_score"`, the global optimum under the configured scorer;
- `rule="minimum_cv_mse"`, the stored component-path row with minimum mean
  response-standardized CV-MSE;
- `rule="one_standard_error"`, the smallest stored component count within one fold-based standard
  error of that minimum.

Each rule retains the predictor rank already selected conditionally for the chosen component count.
With a nondefault scorer, that rank remains conditioned on the scorer even when the component rule
uses response-standardized CV-MSE. The 1-SE rule requires at least two validation splits.

`refit()` clones the configured direct estimator or pipeline, replaces the terminal Pi-PLS rank
pair, fits the clone, and returns it. It does not mutate the search, store the supplied matrices, or
attach the model to search state. Prediction, transformation, scoring, inverse transformation, and
feature-name behavior belong to the returned model.

Post-fit `refit()` and `validation_report()` use the same explicit rule-or-component selection
contract and do not alter search state. The search stores candidate evidence and reusable split
indices, but no selected row, report, or fitted final model.

## Search and model diagnostics

Use `component_path_` for the concise component-count curve and `predictor_rank_profile(h)` for the
evaluated ranks at one count. `cv_results_` contains candidate parameters, split test scores,
response-standardized MSE diagnostics, score ranks, and fit/score timing summaries. It contains only
evaluated candidates.

Use the returned estimator for fixed-model diagnostics such as `decomposition_`, coefficients,
latent scores, predictions, and inspection helpers. Retain the search variable when both search
evidence and the final fitted model are needed.
