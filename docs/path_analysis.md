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

The default `n_components_values="all"` evaluates every admissible component count. An explicit
integer sequence requests a subset:

```python
search = PiPLSPathCV(
    n_components_values=[1, 2, 3, 5],
    refit=False,
).fit(X, Y)
```

Every requested value must have at least one admissible predictor rank.

## Predictor-rank policies

With `predictor_rank_values=None`, predictor rank is selected independently for every component
count. A one-element sequence fixes one rank across the path, a longer sequence defines the
admissible set, and `predictor_rank_values="max"` uses the rule-derived maximum directly.

`search_method="optimal"` evaluates every admissible pair. `search_method="auto"` performs a
deterministic adaptive coarse-to-fine search independently for each component count and may leave
admissible ranks unevaluated. Score ties within numerical tolerance favor the smaller predictor
rank for a fixed component count. The global best then favors the smaller component count and the
smaller predictor rank.

## Rank ceiling

The default upper rank is

\begin{equation}
r_{\pi,\max}=\min\left[p_{\min},n_{\mathrm{train,min}}-1,
r_{\mathrm{num,min}},
\left\lceil\frac{n}{\texttt{samples_per_predictor_rank}}\right\rceil\right].
\end{equation}

Here $n$ is the total number of observations supplied to `fit()`, $p_{\min}$ is the minimum
predictor count after fold-local pipeline preprocessing, and $r_{\mathrm{num,min}}$ is the minimum
verified predictor rank after terminal-estimator centering and optional scaling. The selector fits
pipeline preprocessing separately inside each fold before this rank preflight. An integer
`max_predictor_rank` bypasses the statistical support rule but remains capped by fold dimensions and
numerical rank. Explicit component or predictor-rank values above the resolved ceiling are rejected
before candidate evaluation.

## Scoring and the best evaluated pair

Candidate selection maximizes the configured mean test score. The default scorer is
`pipls.metrics.neg_response_standardized_mean_squared_error`, so maximizing score is equivalent to
minimizing mean response-standardized CV-MSE. With another scorer, the CV-MSE columns remain
diagnostics and need not identify the selected candidate.

Ordinary scikit-learn scorer names, other callables, and `scoring=None` are accepted.
`best_params_`, `best_score_`, and `best_index_` describe the best evaluated pair. Adaptive search
makes no claim about pairs it did not evaluate. The component path remains a model-selection
diagnostic; its numerical minimum does not replace a scientifically justified complexity choice.

## Pipelines and fold-local preprocessing

The supported estimator is either a direct `PiPLSRegression` or a scikit-learn `Pipeline` whose
final step is `PiPLSRegression`. The complete estimator is cloned and fitted inside every fold.

```python
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from pipls import PiPLSPathCV, PiPLSRegression

pipeline = Pipeline(
    [
        ("impute", SimpleImputer()),
        ("pipls", PiPLSRegression(n_components=1, predictor_rank=1)),
    ]
)
search = PiPLSPathCV(estimator=pipeline, refit=False).fit(X, Y)
```

Do not fit learned preprocessing on the complete dataset before path evaluation.

## Cross-validation protocols and metadata

`PiPLSPathCV` accepts scikit-learn splitters for grouped, repeated, predefined, temporal, or
leave-one-out protocols when their scientific assumptions match the data. Suitable examples include
`GroupKFold`, `RepeatedKFold`, `PredefinedSplit`, `TimeSeriesSplit`, and `LeaveOneOut`.

```python
from sklearn.model_selection import GroupKFold

search = PiPLSPathCV(cv=GroupKFold(n_splits=5), refit=False)
search.fit(X, Y, groups=sample_groups)
```

`groups` is an explicit `PiPLSPathCV.fit()` parameter and participates in scikit-learn metadata
routing when routing is enabled and requested. Split metadata belongs to the path selector;
`PiPLSRegression.fit(X, Y)` fits one explicit pair and accepts none.

## Ordered out-of-fold predictions

Set `return_oof_predictions=True` to fit the selected fixed parameterization once per training fold
after selection. The immutable `validation_report_` then owns row-ordered OOF results:

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

## Fold variation

`component_path_.cv_mse_fold_sd` is the population standard deviation of the realized fold-specific
MSE values. It describes fold variation, not a confidence interval. Formal uncertainty statements
require a separately designed repeated or nested resampling procedure.

## Refit and detailed diagnostics

With `refit=True`, the globally best evaluated candidate under the configured scorer is fitted on
all supplied data, and supported prediction or transformation methods delegate to it. The tutorial
uses `refit=False` because it makes the component-count choice visible before fitting one explicit
fixed model.

Use `component_path_` for the concise component-count curve and `predictor_rank_profile(h)` for the
evaluated ranks at one count. `cv_results_` contains candidate parameters, split test scores,
response-standardized MSE diagnostics, score ranks, and fit/score timing summaries. It contains only
evaluated candidates.
