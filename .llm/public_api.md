# Public API contract

## Current top-level API

```python
from pipls import PiPLSRegression
```

`PiPLSPathCV` is planned for a later phase and is not currently exported. Public scoring
callables are available from `pipls.metrics`:

```python
from pipls.metrics import (
    neg_response_standardized_mean_squared_error,
    response_standardized_mean_squared_error,
)
```

## Public names

| Public name | Mathematical notation |
|---|---|
| `n_components` | $h$ |
| `predictor_rank` | $r_\pi$ |
| `samples_per_predictor_rank` | $c$ |
| `predictor_rank_` | fitted $r_\pi$ |
| `max_predictor_rank_` | rule-derived upper bound |

Do not expose constructor aliases named `h`, `r_pi`, or `c`.

## Predictor-rank modes

The estimator supports all three fixed public modes:

```python
PiPLSRegression(n_components=2, predictor_rank=4)
PiPLSRegression(n_components=2, predictor_rank="max")
PiPLSRegression(n_components=2, predictor_rank="auto")
```

The default is `predictor_rank="auto"`. Automatic mode materializes one CV split set, computes
its smallest training-fold size, searches every integer rank from `n_components` through the
fold-safe upper bound, selects the largest scikit-learn score with deterministic low-rank
tie-breaking, and refits the selected fixed-rank model on all data passed to `fit()`.

## Accepted search-policy transition

The exhaustive behavior above is the currently implemented Phase C2b behavior, but its public
name is provisional. Decision 0007 establishes the target semantics for the next implementation
increment:

- `predictor_rank="optimal"` will perform the exhaustive scan currently called `"auto"`;
- `predictor_rank="auto"` will perform a deterministic adaptive coarse-to-fine search and may
  evaluate only a subset of admissible ranks;
- `predictor_rank="max"` and explicit integer ranks remain unchanged.

Until Phase C2c is implemented, source code and tests remain authoritative for runtime behavior.
The adaptive mode must expose which ranks were evaluated and whether its result was exhaustive.
Randomized SVD is not part of the rank-mode meaning and will be governed by a separate solver
parameter in a later increment.

The upper bound is

\begin{equation}
r_{\pi,\max}
=
\min\left[
p,
n_{\mathrm{train,min}},
\left\lceil
\frac{n_{\mathrm{train,min}}}{\texttt{samples\_per\_predictor\_rank}}
\right\rceil
\right].
\end{equation}

For `"auto"`, `n_train_min` is derived from the materialized internal-CV splits. For `"max"`,
the samples supplied to `fit()` are the training data. An explicit integer bypasses the
rule-derived bound but remains subject to the core numerical-rank and dimensional checks.

## Internal search parameters

- `cv` accepts an integer, a scikit-learn splitter, or an iterable of `(train, validation)` pairs.
  The default `cv=5` is deterministic, unshuffled regression K-fold splitting through
  scikit-learn's `check_cv` behavior.
- `scoring` accepts a scikit-learn scorer name or scorer callable. The default is
  `"neg_response_standardized_mean_squared_error"`.
- `n_jobs` controls parallel evaluation across predictor-rank candidates through joblib.

The positive response-standardized utility follows the scorer signature `(estimator, X, y)`.
The negative version is suitable for scikit-learn search APIs, where larger scores are better.

## Fitted estimator behavior

The estimator provides `fit`, `predict`, `transform`, and scalar `score`. It exposes
`predictor_rank_`, `max_predictor_rank_`, preprocessing statistics, Pi-PLS factorization arrays,
latent scores, `coef_` in scikit-learn orientation, `coef_matrix_` in manuscript orientation, and
`intercept_`.

Automatic mode additionally exposes:

- `predictor_rank_values_`;
- `predictor_rank_cv_results_` with split and mean scores plus response-standardized MSE;
- `best_score_`;
- `best_response_standardized_mse_` for the selected rank;
- `n_splits_`;
- `cv_n_train_min_`.

`response_scale_for_scoring_` is estimated from the data used to fit each estimator with
`ddof=1`, independently of whether `scale` is true or false.
