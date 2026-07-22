# Advanced path-search behavior

The [synthetic tutorial](tutorials/synthetic.md#evaluate-the-component-path) owns the routine
selection workflow. This page records configuration and edge behavior needed when the defaults are not enough.
Exact signatures and fitted attributes are in the [generated path API](api/path.md).

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

```python
fixed_rank_path = PiPLSPathCV(
    n_components_values=[1, 2, 3],
    predictor_rank_values=[8],
    refit=False,
).fit(X, Y)
```

`search_method="optimal"` evaluates every admissible pair. `search_method="auto"` performs a
deterministic adaptive coarse-to-fine search independently for each component count and may leave
admissible ranks unevaluated. Score ties within numerical tolerance favor the smaller predictor
rank for a fixed component count. The global best then favors the smaller component count and the
smaller predictor rank.

## Rank ceiling

The default upper rank is

\begin{equation}
r_{\pi,\max}=\min\left[p_{\min},n_{\mathrm{train,min}}-1,
\left\lceil\frac{n}{\texttt{samples_per_predictor_rank}}\right\rceil\right].
\end{equation}

Here $n$ is the total number of observations supplied to `fit()`, while fold dimensions impose hard
feasibility limits. An integer `max_predictor_rank` bypasses the statistical support rule but remains
capped by centered fold-feasible dimensions.

## Scoring and the best evaluated pair

Candidate selection maximizes the configured mean test score. The default scorer is
`pipls.metrics.neg_response_standardized_mean_squared_error`, so maximizing score is equivalent to
minimizing mean response-standardized CV-MSE. With another scorer, the CV-MSE columns remain
diagnostics and need not identify the selected candidate.

`best_params_`, `best_score_`, and `best_index_` describe the best evaluated pair. Adaptive search
makes no claim about pairs it did not evaluate. The component path remains a model-selection
diagnostic; its numerical minimum does not replace a scientifically justified complexity choice.

## Pipelines and fold-local preprocessing

The supported estimator is either a direct `PiPLSRegression` or a scikit-learn `Pipeline` whose
final step is `PiPLSRegression`. The complete estimator is cloned and fitted inside every fold.

```python
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

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

## Refit and detailed diagnostics

With `refit=True`, the globally best evaluated candidate under the configured scorer is fitted on
all supplied data, and supported prediction or transformation methods delegate to it. The tutorial
uses `refit=False` because it makes the component-count choice visible before fitting one explicit
fixed model.

Use `component_path_` for the concise component-count curve and `predictor_rank_profile(h)` for the
evaluated ranks at one count. The complete candidate-level arrays, split values, timings, resolved
grids, and search history remain in `cv_results_`. No duplicate matrix-shaped path surface is
stored.

Group-aware and advanced splitters are accepted. Set `return_oof_predictions=True` only when
row-aligned OOF predictions for the selected evaluated pair are required. See
[Cross-validation protocols and OOF reporting](cross_validation.md).
