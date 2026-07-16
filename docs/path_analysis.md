# Pi-PLS path analysis

`PiPLSPathCV` evaluates the admissible two-parameter Pi-PLS surface

\[
\mathcal{G}=\{(h,r_\pi):1\le h\le h_{\max},\ h\le r_\pi\le r_{\pi,\max}\}.
\]

It is the preferred interface when the complete conditional path over
`n_components` is required or when learned preprocessing must be fitted inside every
rank-selection fold.

## Basic use

```python
from pipls import PiPLSPathCV

search = PiPLSPathCV(
    samples_per_predictor_rank=10,
    cv=5,
    search_method="optimal",
    n_jobs=-1,
)
search.fit(X, Y)

print(search.best_params_)
print(search.best_predictor_rank_by_n_components_)
Y_pred = search.predict(X_new)
```

`search_method="optimal"` is the default and evaluates every admissible pair.
`search_method="auto"` performs deterministic logarithmic coarse-to-fine predictor-rank
search independently for each `n_components` value and may skip candidates.

The default fold-safe upper rank is

\[
r_{\pi,\max}=\min\left[p_{\min},n_{\mathrm{train,min}},
\left\lceil\frac{n_{\mathrm{train,min}}}
{\texttt{samples_per_predictor_rank}}\right\rceil\right],
\]

where `p_min` is the smallest predictor dimension reaching the Pi-PLS step across
training folds. Supplying an integer `max_predictor_rank` bypasses the statistical rule but
remains capped by fold-safe algebraic dimensions.

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
    cv=10,
)
search.fit(X, Y)
```

The path estimator clones and fits the entire pipeline separately for every fold and
candidate. It infers a unique nested `PiPLSRegression` step. For deeper composites, use
`pipls_param_prefix`, for example `"regressor__regression"`.

## Selection and diagnostics

The global selection rule maximizes the configured scikit-learn score. Ties within the
package tolerance prefer smaller `n_components` and then smaller `predictor_rank`.
Conditional ties for fixed `n_components` prefer smaller predictor rank.

Important fitted attributes include:

- `cv_results_`, `best_params_`, `best_score_`, and `best_estimator_`;
- `best_n_components_` and `best_predictor_rank_`;
- `best_predictor_rank_by_n_components_` and `best_score_by_n_components_`;
- `response_standardized_mse_path_` and `score_path_`;
- `n_components_values_`, `predictor_rank_values_`, and `max_predictor_rank_`;
- `n_path_candidates_`, `n_path_candidates_evaluated_`, and
  `n_path_candidates_skipped_`;
- `path_search_method_`, `path_search_history_`, and `path_search_exhaustive_`.

When `refit=True`, the selected complete estimator is fitted once on all supplied data.
With `refit=False`, path diagnostics remain available but `predict`, `transform`, and
`score` are disabled.
