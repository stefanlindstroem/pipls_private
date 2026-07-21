# Parameter selection

Pi-PLS uses paired predictor and response latent variables. `n_components` controls how many pairs
are retained, while `predictor_rank` controls how much predictor variation is available when those
pairs are estimated. See the [theory overview](theory.md#interpretation-of-the-ranks) for the
mathematical distinction.

The usual workflow scans candidate component counts by cross-validation. For each count,
`PiPLSPathCV` selects a predictor rank and reports cross-validated mean squared error (CV-MSE). The
resulting table or curve is the **component path**. Inspect CV-MSE against the number of components
and choose a parsimonious point, often the elbow or plateau where additional components give little
improvement. The absolute
minimum is informative, but it need not be the final scientific choice.

Pi-PLS separates this selection step from fixed-model fitting:

- `PiPLSPathCV` evaluates the bounded two-parameter search;
- `PiPLSRegression` fits one explicit pair `(n_components, predictor_rank)`.

## Evaluate the component path

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
scorer, this is equivalent to minimizing mean response-standardized CV-MSE. After inspecting the
curve, choose a component count and fit both ranks explicitly:

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
The reported fold SD describes fold-to-fold variation and is not a confidence interval.

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
