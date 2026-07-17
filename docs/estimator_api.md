# `PiPLSRegression`

The estimator follows the scikit-learn fit/predict interface. Ordinary use relies on adaptive
conditional predictor-rank selection:

```python
from pipls import PiPLSRegression

model = PiPLSRegression(n_components=2)
model.fit(X_train, Y_train)
Y_pred = model.predict(X_test)
print(model.predictor_rank_)
```

Constructor parameters are `n_components`, `scale`, `copy`, `predictor_rank`,
`samples_per_predictor_rank`, `cv`, `scoring`, `n_jobs`, `svd_solver`, and `random_state`.
`predictor_rank` accepts `"auto"`, `"optimal"`, `"max"`, or a positive integer. The ordinary
selection defaults are `samples_per_predictor_rank=5` and `cv=5`.

`"optimal"` exhaustively evaluates every admissible rank. `"auto"` uses deterministic
logarithmic coarse-to-fine search and may skip ranks; it becomes exhaustive when the admissible or
final refinement interval contains at most 10 ranks. Both modes reuse one materialized CV split
set, fit preprocessing inside each training fold, apply identical scoring and low-rank tie rules,
derive the samples-per-rank support term from all supplied observations, and refit the selected
rank on all data. Smaller centered training folds remain hard feasibility caps.

`svd_solver="full"` uses the exact thin predictor SVD. `svd_solver="randomized"` uses a
reproducible randomized truncated predictor SVD. The default `svd_solver="auto"` chooses randomized
SVD only for large matrices and low retained-rank fractions; the fitted choice is available as
`svd_solver_`. Only the predictor SVD may be randomized.

`scale=True` centers and divides predictor and response columns by training-sample standard
deviations with `ddof=1`. `scale=False` centers without division. Constant columns use scale 1.
Every learned preprocessing statistic is fitted independently inside selection folds.

## PLS-style estimator surface

`predict(X, copy=True)`, `transform(X, y=None, copy=True)`, `fit_transform(X, y)`, and
`inverse_transform(X, y=None)` mirror the corresponding `PLSRegression` conventions.
`transform(X)` returns predictor scores; supplying `y` returns `(x_scores, y_scores)`.
`inverse_transform` reconstructs original-unit predictors and, when supplied, responses through the
least-squares loading matrices; reconstruction is approximate unless the retained spaces span the
centered/scaled data. The estimator supports `feature_names_in_`, `get_feature_names_out()`, and
`set_output(transform="pandas")`.

Standard fitted attributes are `x_weights_`, `y_weights_`, `x_loadings_`, `y_loadings_`,
`x_scores_`, `y_scores_`, `x_rotations_`, `y_rotations_`, `coef_`, and `intercept_`. Because Pi-PLS
uses direct orthogonal score maps rather than iterative deflation, weights and rotations coincide.
Loadings are separate least-squares reconstruction coefficients. `n_iter_` is deliberately absent.

Pi-PLS-specific factorization output is available through the public frozen
`PiPLSDecomposition` instance at `decomposition_`. It contains `Pi`, `C`, `W`, `P`, `D`, `Q`, the
dilation vector, numerical-rank diagnostics, and the resolved predictor SVD solver. The existing
`Pi_`, `C_`, `W_`, `P_`, `D_`, `Q_`, and `dilation_` are read-only identity aliases to the
canonical arrays stored in `decomposition_`, preventing divergent factorization state.

Important additional fitted attributes include `predictor_rank_`, `max_predictor_rank_`,
`svd_solver_`, `x_rank_is_exact_`, `x_mean_`, `x_scale_`, `y_mean_`, `y_scale_`, and
`response_scale_for_scoring_`. Cross-validated modes expose standard `cv_results_`, `best_params_`,
`best_index_`, `best_score_`, and `scorer_` attributes plus candidate scores, standard fit/score
timing columns, evaluation order, search
batches, the final refinement interval, candidate counts, whether the search was exhaustive, and
the minimum CV training-fold size used for feasibility. `predictor_rank_cv_results_` is an alias
for `cv_results_`.

The constructor `copy` controls fit-time preprocessing. For writable floating NumPy arrays,
`copy=False` permits in-place centering and scaling, matching the familiar PLS contract.


## Parameter validation

Integer controls are strict. `n_components` and an integer `predictor_rank` must be positive
integers; integer `cv` must be at least 2, while `cv=None` requests standard five-fold regression
CV; `scoring=None` uses estimator `score`; `n_jobs` must be `None` or nonzero; and
`random_state` must lie in the unsigned 32-bit interval. Booleans are not accepted as integers,
and integral-valued floats are not silently converted. NumPy integer scalars are accepted.

`random_state=None` is valid only with `svd_solver="full"`. The automatic solver may choose
randomized SVD and therefore requires a reproducible seed.

For rule-based rank modes, `samples_per_predictor_rank < 5` emits
`pipls.StatisticalSupportWarning`. Such settings are permitted, but the resulting rank bound may
not have sufficient statistical support to be trusted without external validation. An explicit
integer rank bypasses this rule and does not emit the warning.


## Cross-validation metadata and reporting

The fit signature accepts keyword-only `groups` for group-aware internal CV.
`return_oof_predictions=True` requests an additional fixed-parameter OOF pass after selection and
attaches `PiPLSValidationReport` at `validation_report_`. For repeated splitters, predictions are
averaged; for partial-coverage splitters, uncovered rows contain NaN.
