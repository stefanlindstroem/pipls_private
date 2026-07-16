# Public API contract

## Current top-level API

```python
from pipls import PiPLSRegression
```

`PiPLSPathCV` is planned for a later phase and is not currently exported.

## Public names

| Public name | Mathematical notation |
|---|---|
| `n_components` | $h$ |
| `predictor_rank` | $r_\pi$ |
| `samples_per_predictor_rank` | $c$ |
| `predictor_rank_` | fitted $r_\pi$ |
| `max_predictor_rank_` | rule-derived upper bound |

Do not expose constructor aliases named `h`, `r_pi`, or `c`.

## Current rank modes

The currently implemented constructor supports an explicit positive integer and the rule-fixed
mode:

```python
PiPLSRegression(n_components=2, predictor_rank=4)
PiPLSRegression(
    n_components=2,
    predictor_rank="max",
    samples_per_predictor_rank=10,
)
```

For `predictor_rank="max"`, fitting on $n_{\mathrm{train}}$ samples and $p$ predictor columns
uses

\begin{equation}
r_{\pi,\max}
=
\min\left[
p,
n_{\mathrm{train}},
\left\lceil
\frac{n_{\mathrm{train}}}{\texttt{samples\_per\_predictor\_rank}}
\right\rceil
\right].
\end{equation}

The data passed to `fit()` are the training data for this non-CV mode. A later automatic-selection
phase will use the same helper with the smallest materialized internal-CV training-fold size.
An explicit integer rank bypasses the rule-derived bound but remains subject to the core numerical
rank and dimensional admissibility checks.

`predictor_rank="auto"` remains planned and must not be accepted until fold-local automatic
selection is implemented.

## Fitted estimator behavior

The estimator provides `fit`, `predict`, `transform`, and scalar `score`. It exposes
`predictor_rank_`, `max_predictor_rank_`, preprocessing statistics, Pi-PLS factorization arrays,
latent scores, `coef_` in scikit-learn orientation, `coef_matrix_` in manuscript orientation, and
`intercept_`.
