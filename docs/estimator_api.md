# `PiPLSRegression`

The current estimator follows the scikit-learn fit/predict interface:

```python
from pipls import PiPLSRegression

model = PiPLSRegression(
    n_components=2,
    predictor_rank="max",
    samples_per_predictor_rank=10,
)
model.fit(X_train, Y_train)
Y_pred = model.predict(X_test)
```

Implemented constructor parameters are `n_components`, `scale`, `copy`, `predictor_rank`, and
`samples_per_predictor_rank`. The current `predictor_rank` values are a positive integer or
`"max"`; `"auto"` remains reserved for a later implementation phase.

`scale=True` centers and divides predictor and response columns by training-sample standard
deviations with `ddof=1`. `scale=False` centers without division. Constant columns use scale 1.

Important fitted attributes include `predictor_rank_`, `max_predictor_rank_`, `x_mean_`,
`x_scale_`, `y_mean_`, `y_scale_`, the Pi-PLS factorization arrays, `coef_`, `coef_matrix_`, and
`intercept_`.
