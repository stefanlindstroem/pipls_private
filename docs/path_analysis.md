# Path-search reference

`PiPLSPathCV` evaluates component count and predictor rank within one bounded search. It is the
selection object; `PiPLSRegression` remains a fixed-parameter estimator. For the concise procedure,
see [Parameter selection](parameter_selection.md). For a complete worked analysis, see the
[Pulp tutorial](tutorials/pulp.md#evaluate-the-component-path).

## Search surface

For component count $h$ and predictor rank $r_\pi$, the admissible pairs are

\begin{equation}
\mathcal{G}=\{(h,r_\pi):1\le h\le h_{\max},\ h\le r_\pi\le r_{\pi,\max}\}.
\end{equation}

The default path evaluates all admissible component counts and selects one predictor rank
conditionally for each count:

```python
from pipls import PiPLSPathCV

search = PiPLSPathCV(refit=False).fit(X, Y)
```

`component_path_` is the immutable concise view. Its aligned read-only arrays are:

```text
n_components
predictor_rank
predictor_rank_policy
mean_test_score
cv_mse_mean
cv_mse_fold_sd
n_splits
```

Use `component_path_.for_n_components(h)` to retrieve one frozen scalar result for an evaluated
component count.

## Component-count requests

`n_components_values="all"` evaluates every admissible component count. An explicit integer
sequence evaluates only those values:

```python
search = PiPLSPathCV(
    n_components_values=[1, 2, 3, 5],
    refit=False,
).fit(X, Y)
```

Every requested value must have at least one admissible predictor rank.

## Predictor-rank policies

With `predictor_rank_values=None`, the path selects predictor rank independently for every component
count. A one-element sequence fixes one rank across the path. A longer sequence defines the
admissible set. `predictor_rank_values="max"` uses the rule-derived maximum directly.

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
rank for a fixed component count. The global best among evaluated pairs favors the smaller
component count and then the smaller predictor rank.

## Rank ceiling

The default upper rank is

\begin{equation}
r_{\pi,\max}=\min\left[p_{\min},n_{\mathrm{train,min}}-1,
\left\lceil\frac{n}{\texttt{samples_per_predictor_rank}}\right\rceil\right].
\end{equation}

Here $n$ is the total number of observations supplied to `fit()`, while fold dimensions impose hard
feasibility limits. An integer `max_predictor_rank` bypasses the statistical support rule but remains
capped by centered fold-feasible dimensions.

## Scoring and best evaluated parameters

Candidate selection maximizes the configured mean test score. The default scorer is
`pipls.metrics.neg_response_standardized_mean_squared_error`, so maximizing score is equivalent to
minimizing mean response-standardized CV-MSE. With another scorer, the CV-MSE columns remain
diagnostics and need not identify the selected candidate.

`best_params_`, `best_score_`, and `best_index_` describe the best **evaluated** pair. Adaptive search
makes no claim about pairs it did not evaluate. The path curve remains a model-selection diagnostic;
its numerical minimum does not replace the user's final complexity choice.

## Complete-pipeline search

The supported estimator is either a direct `PiPLSRegression` or a scikit-learn `Pipeline` whose
final step is `PiPLSRegression`. The complete estimator is cloned and fitted inside every fold.

```python
from sklearn.pipeline import Pipeline

from pipls import PiPLSPathCV, PiPLSRegression

pipeline = Pipeline(
    [("pipls", PiPLSRegression(n_components=1, predictor_rank=1))]
)
search = PiPLSPathCV(
    estimator=pipeline,
    n_components_values=[1, 2, 3],
    refit=False,
).fit(X, Y)
```

Any learned preprocessing placed before the final Pi-PLS step is therefore fitted within each
training fold. Do not preprocess the complete dataset before path evaluation.

## Refit and diagnostics

With `refit=True`, the best evaluated estimator is fitted on all supplied data and supported
prediction or transformation methods delegate to it. With `refit=False`, the path diagnostics
remain available without a fitted final estimator.

Use `predictor_rank_profile(h)` for the one-dimensional candidate profile at an evaluated component
count. It returns only ranks actually evaluated by the fitted search, sorted in ascending order,
together with aligned mean scores, CV-MSE summaries, fold standard deviations, and the conditionally
selected scalar row. The method derives this immutable view on demand; it does not add another fitted
representation that must remain synchronized.

The complete evaluated search surface remains available through the aligned candidate arrays in
`cv_results_`, together with resolved rank grids, candidate counts, and search history.
Matrix-shaped score aliases are intentionally not duplicated: advanced analysis can reshape the
`cv_results_` columns when a dense surface is useful.
See the [generated path API](api/path.md) for exact attributes and conditional availability.

## Splitters and OOF predictions

Group-aware and advanced scikit-learn splitters are accepted. Set `return_oof_predictions=True`
only when row-aligned OOF predictions for the selected evaluated pair are required. Those
predictions are selection-conditioned because the same path search selected the parameters. See
[Cross-validation protocols and OOF reporting](cross_validation.md).
