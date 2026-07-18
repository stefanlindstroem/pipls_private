# Parameter selection

Pi-PLS separates fixed-model fitting from model selection.

- `PiPLSRegression` fits one explicit pair `(n_components, predictor_rank)`.
- `PiPLSPathCV` is the standard package workflow for the bounded triangular search.

## Component-path workflow

```python
import pandas as pd

from pipls import PiPLSPathCV, PiPLSRegression

search = PiPLSPathCV(refit=False).fit(X, Y)

path = pd.DataFrame(search.component_path_results_)
path.to_csv("component_path.csv", index=False)
```

The default `n_components_values="all"` evaluates every admissible component count. An explicit
integer sequence requests a smaller or nonconsecutive path.

For each component count, the path selects predictor rank by maximizing the configured mean CV
score and reports the corresponding response-standardized CV-MSE. Under the default negative-MSE
scorer, this is equivalent to minimizing mean response-standardized CV-MSE. After inspecting that
path, the user chooses a component count and fits both ranks explicitly:

```python
chosen_n_components = 3
path = pd.read_csv("component_path.csv").set_index("n_components")
chosen_predictor_rank = int(path.loc[chosen_n_components, "predictor_rank"])

model = PiPLSRegression(
    n_components=chosen_n_components,
    predictor_rank=chosen_predictor_rank,
).fit(X, Y)
```

`PiPLSPathCV.best_params_` identifies the best evaluated pair under the configured scorer. With the
default scorer, it has the smallest evaluated mean response-standardized CV-MSE. Under adaptive
`search_method="auto"`, admissible pairs that were not evaluated are not part of that comparison.
The numerical selection is not presented as a mandatory scientific choice.

## Predictor-rank ceiling

The default path ceiling is

\[
r_{\pi,\max}
=
\min\left[
 p_{\min},
 n_{\mathrm{train,min}}-1,
 \left\lceil\frac{n}{c}\right\rceil
\right],
\qquad c=5.
\]

The support term uses the total number of observations supplied to `fit()`. Fold dimensions only
impose feasibility caps. `search_method="auto"` performs deterministic adaptive coarse-to-fine
search and may leave admissible predictor ranks unevaluated; `search_method="optimal"` evaluates all
admissible pairs. Score ties within numerical tolerance favor the smaller predictor rank for a
fixed component count. The global best among evaluated pairs favors the smaller component count,
then the smaller predictor rank.

The canonical `component_path_results_` columns are:

```text
n_components
predictor_rank
predictor_rank_policy
response_standardized_cv_mse_mean
response_standardized_cv_mse_fold_sd
n_splits
```

The predictor rank is always numeric, whether it was optimized, fixed, or set to the maximum.
