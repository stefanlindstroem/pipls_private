# Pi-PLS path analysis

`PiPLSPathCV` evaluates the admissible two-parameter Pi-PLS surface

\[
\mathcal{G}=\{(h,r_\pi):1\le h\le h_{\max},\ h\le r_\pi\le r_{\pi,\max}\}.
\]

The ordinary user workflow is deliberately two-stage:

1. evaluate one conditional predictor-rank result for each candidate `n_components` value;
2. inspect the resulting CV-MSE path and fit a separate fixed model with the chosen pair.

With the default scorer, the numerically smallest evaluated CV-MSE is informative, but it is not
treated as an automatic scientific choice of model complexity.

## Stage 1: evaluate the component path

```python
import pandas as pd

from pipls import PiPLSPathCV

search = PiPLSPathCV(refit=False).fit(X, Y)

component_path = pd.DataFrame(search.component_path_results_)
component_path.to_csv("component_path.csv", index=False)
```

The default `n_components_values="all"` resolves to every admissible component count from 1
through `min(n_targets, max_predictor_rank_)`. Supply an explicit sequence when the scientific
question concerns only a subset:

```python
search = PiPLSPathCV(
    n_components_values=[1, 2, 3, 4],
    refit=False,
).fit(X, Y)
```

`component_path_results_` contains one row per requested component count:

```text
n_components
predictor_rank
predictor_rank_policy
response_standardized_cv_mse_mean
response_standardized_cv_mse_fold_sd
n_splits
```

The numeric `predictor_rank` is always present. The policy column has one of three values:

- `optimized`: predictor rank was selected conditionally for that component count;
- `fixed`: one explicit predictor rank was used;
- `maximum`: the rule-derived maximum predictor rank was used directly.

The fold SD is the standard deviation of the fold-specific response-standardized MSE values. It is
a descriptive measure of fold-to-fold variation, not a confidence interval or an independent
standard error, because cross-validation training sets overlap. Conditional rows are selected by
the configured scorer. With a nondefault scorer, the reported MSE remains a diagnostic and need not
be the quantity minimized by selection.

### Predictor-rank policies

The default conditionally optimizes predictor rank independently for every admissible component
count:

```python
search = PiPLSPathCV(refit=False).fit(X, Y)
```

A one-element sequence fixes one predictor rank across the path:

```python
search = PiPLSPathCV(
    n_components_values=[1, 2, 3],
    predictor_rank_values=[8],
    refit=False,
).fit(X, Y)
```

A longer sequence defines the admissible predictor-rank set. `search_method="optimal"` evaluates
the complete set, while `"auto"` may evaluate only an adaptive subset. `"max"` uses the rule-derived
maximum directly:

```python
search = PiPLSPathCV(
    n_components_values=[1, 2, 3, 4],
    predictor_rank_values="max",
    refit=False,
).fit(X, Y)
```

Every requested component count must have at least one admissible predictor rank satisfying
`n_components <= predictor_rank`.

## Stage 2: fit the chosen fixed model

After inspecting the CSV or a plot derived from it, choose one component count and read the
matching predictor rank from the table:

```python
import pandas as pd

from pipls import PiPLSRegression

component_path = pd.read_csv("component_path.csv").set_index("n_components")
chosen_n_components = 3
chosen_predictor_rank = int(
    component_path.loc[chosen_n_components, "predictor_rank"]
)

model = PiPLSRegression(
    n_components=chosen_n_components,
    predictor_rank=chosen_predictor_rank,
).fit(X, Y)
```

Both ranks are fixed in the final fit. This reproduces the parameterization represented by the
chosen path row rather than performing a second automatic rank search.

The repository real-data examples use the same sequence directly in Python. They evaluate the
standard-PLS comparison with an imported helper function, write both CSV files, call the plotting
function on those files, and then read the chosen Pi-PLS row. No subprocess or hidden dataset I/O
layer is involved.

## Search settings and rank limits

`n_components_values="all"`, `search_method="auto"`, `samples_per_predictor_rank=5`, and `cv=5`
are the default search settings. Set `search_method="optimal"` to evaluate every admissible pair.
`search_method="auto"` performs deterministic logarithmic coarse-to-fine predictor-rank search
independently for each `n_components` value and may skip candidates.

The default full-sample-supported, fold-feasible upper rank is

\[
r_{\pi,\max}=\min\left[p_{\min},n_{\mathrm{train,min}}-1,
\left\lceil\frac{n}{\texttt{samples_per_predictor_rank}}\right\rceil\right],
\]

where $n$ is the total number of observations supplied to `fit()` and $p_{\min}$ is the smallest
predictor dimension reaching the Pi-PLS step across training folds. Supplying an integer
`max_predictor_rank` bypasses the statistical rule but remains capped by centered fold-feasible
dimensions.

## Complete-pipeline search

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from pipls import PiPLSPathCV, PiPLSRegression

pipeline = Pipeline([
    ("preprocess", StandardScaler()),
    ("regression", PiPLSRegression(
        n_components=1,
        predictor_rank=1,
        scale=False,
    )),
])

search = PiPLSPathCV(
    estimator=pipeline,
    n_components_values=[1, 2, 3],
    refit=False,
    cv=10,
).fit(X, Y)
```

The supported estimator forms are either a direct `PiPLSRegression` or a scikit-learn `Pipeline`
whose final step is `PiPLSRegression`. The complete supported estimator is cloned and fitted
separately for every fold and candidate. The terminal Pi-PLS step is inferred from the supported
estimator structure; no separate parameter-prefix control is required.

## Complete search diagnostics

`component_path_results_` is the concise user-facing view. The full search surface remains available
through:

- `cv_results_`, `best_params_`, `best_score_`, and `best_index_`;
- `best_predictor_rank_by_n_components_` and `best_score_by_n_components_`;
- `response_standardized_mse_path_` and `score_path_`;
- `n_components_values_`, `predictor_rank_values_`, `predictor_rank_policy_`, and
  `max_predictor_rank_`;
- candidate-count, search-history, and exhaustive-search diagnostics.

`best_params_` identifies the best evaluated pair by maximum mean test score. With the default
negative-MSE scorer, this is the evaluated pair with the smallest mean response-standardized
CV-MSE. Adaptive search makes no claim about admissible pairs it did not evaluate. This numerical
selection does not replace the user decision shown in the two-stage examples.

When `refit=True`, the best evaluated estimator is fitted on all supplied data and supported
prediction or transformation methods delegate to it. With `refit=False`, those methods are absent
and all path diagnostics remain available. Output-container configuration belongs to the estimator
template; configuring a direct estimator before passing it to the path is preserved through
candidate cloning and the selected refit.

Direct fixed fits warn when they have fewer than four observations per retained predictor-rank
direction. `PiPLSPathCV` suppresses only that expected diagnostic for its controlled feature
probes, candidate folds, optional OOF fits, and selected full-data refit. Other warnings from a
pipeline or estimator remain visible.

## Advanced splitters and OOF output

`fit(X, y, groups=groups)` supports group-aware splitters. Repeated, predefined, temporal, and
leave-one-out protocols use their ordinary scikit-learn splitter objects. Set
`return_oof_predictions=True` only when row-ordered OOF predictions for the best evaluated
parameter pair are specifically required. Those diagnostics are explicitly selection-conditioned.
See `cross_validation.md`.
