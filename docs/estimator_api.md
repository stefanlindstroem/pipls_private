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
`samples_per_predictor_rank`, `cv`, `scoring`, and `n_jobs`. `predictor_rank` accepts `"auto"`,
`"optimal"`, `"max"`, or a positive integer.

`"optimal"` exhaustively evaluates every admissible rank. `"auto"` uses deterministic
logarithmic coarse-to-fine search and may skip ranks; it becomes exhaustive when the admissible or
final refinement interval contains at most 10 ranks. Both modes reuse one materialized CV split
set, fit preprocessing inside each training fold, apply identical scoring and low-rank tie rules,
and refit the selected rank on all data.

`scale=True` centers and divides predictor and response columns by training-sample standard
deviations with `ddof=1`. `scale=False` centers without division. Constant columns use scale 1.
Every learned preprocessing statistic is fitted independently inside selection folds.

Important fitted attributes include `predictor_rank_`, `max_predictor_rank_`, `x_mean_`,
`x_scale_`, `y_mean_`, `y_scale_`, `response_scale_for_scoring_`, the Pi-PLS factorization arrays,
`coef_`, `coef_matrix_`, and `intercept_`. Cross-validated modes also expose candidate scores,
evaluation order, search batches, the final refinement interval, candidate counts, whether the
search was exhaustive, the selected score, and the fold-safe training-size bound.
