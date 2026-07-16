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
    n_jobs=-1,
)
search.fit(X, Y)

print(search.best_params_)
print(search.best_pipls_params_)
print(search.best_predictor_rank_by_n_components_)
Y_pred = search.predict(X_new)
```

`search_method="auto"` is the default and performs adaptive search. Set
`search_method="optimal"` to evaluate every admissible pair.
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

The supported estimator forms are deliberately explicit: either a direct `PiPLSRegression`, or a
scikit-learn `Pipeline` whose final step is `PiPLSRegression`. The path estimator preserves
indexable input containers, then clones and fits the entire supported pipeline separately for every
fold and candidate. This permits pandas column names and name-based `ColumnTransformer` selectors
to remain available inside every fold. `pipls_param_prefix` may name the final Pi-PLS pipeline step;
it is otherwise inferred. Arbitrary nested meta-estimators are rejected rather than partially
supported.

## Selection and diagnostics

The global selection rule maximizes the configured scikit-learn score. Ties within the
package tolerance prefer smaller `n_components` and then smaller `predictor_rank`.
Conditional ties for fixed `n_components` prefer smaller predictor rank.

Important fitted attributes include:

- `cv_results_`, `best_params_`, `best_score_`, `scorer_`, and `best_estimator_`;
- standard `mean_fit_time`, `std_fit_time`, `mean_score_time`, and `std_score_time` columns;
- `refit_time_` when `refit=True`;
- `best_pipls_`, the selected fitted nested `PiPLSRegression`, and `best_pipls_params_`;
- `best_n_components_` and `best_predictor_rank_`;
- `best_predictor_rank_by_n_components_` and `best_score_by_n_components_`;
- `response_standardized_mse_path_` and `score_path_`;
- `n_components_values_`, `predictor_rank_values_`, and `max_predictor_rank_`;
- `n_path_candidates_`, `n_path_candidates_evaluated_`, and
  `n_path_candidates_skipped_`;
- `path_search_method_`, `path_search_history_`, and `path_search_exhaustive_`.

When `refit=True`, the selected complete estimator is fitted once on all supplied data. Standard
`predict`, conditional `transform`/`fit_transform`/`inverse_transform`, feature-name, pandas-output,
and R2 `score` behavior delegates to the selected estimator. Methods are exposed only when the
selected estimator supports them. `best_score_` remains the configured selection score and can
differ from the R2 returned by `score`. `cv=None` requests standard five-fold regression CV and
`scoring=None` uses the estimator's own `score` method.

With `refit=False`, path diagnostics remain available but `predict`, `transform`, and
`score` are disabled.


## Advanced splitters and OOF output

`fit(X, y, groups=groups)` supports group-aware splitters. Repeated, predefined, temporal, and
leave-one-out protocols use their ordinary scikit-learn splitter objects. Set
`return_oof_predictions=True` to expose `oof_predictions_`, `oof_prediction_counts_`,
`oof_params_`, `pooled_oof_r2_`, and the immutable `validation_report_`. Repeated predictions are
averaged; uncovered rows remain NaN. Path validation reports are explicitly
`selection-conditioned`. See `cross_validation.md`.
