# `PiPLSRegression`

The estimator follows the scikit-learn fit/predict interface. Ordinary use relies on automatic
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
`"max"`, or a positive integer.

`scale=True` centers and divides predictor and response columns by training-sample standard
deviations with `ddof=1`. `scale=False` centers without division. Constant columns use scale 1.
Every learned preprocessing statistic is fitted independently inside automatic-selection folds.

Important fitted attributes include `predictor_rank_`, `max_predictor_rank_`, `x_mean_`,
`x_scale_`, `y_mean_`, `y_scale_`, `response_scale_for_scoring_`, the Pi-PLS factorization arrays,
`coef_`, `coef_matrix_`, and `intercept_`. Automatic mode also exposes the candidate ranks,
per-split and mean CV diagnostics, the selected score, and the fold-safe training-size bound.
