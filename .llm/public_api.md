# Public API contract

## Planned top-level API

```python
from pipls import PiPLSPathCV, PiPLSRegression
```

The estimator is not implemented in the repository seed.

## Planned public names

| Public name | Mathematical notation |
|---|---|
| `n_components` | $h$ |
| `predictor_rank` | $r_\pi$ |
| `samples_per_predictor_rank` | $c$ |
| `predictor_rank_` | selected $r_\pi$ |
| `max_predictor_rank_` | rule-derived upper bound |

Do not expose constructor aliases named `h`, `r_pi`, or `c`.

The principal future call is:

```python
model = PiPLSRegression(n_components=3)
model.fit(X, Y)
Y_pred = model.predict(X_new)
```
