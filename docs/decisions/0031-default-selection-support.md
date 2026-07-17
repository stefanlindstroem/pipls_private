# Decision 0031: default selection support and fold count

## Status

Accepted and implemented.

## Context

The public rule-derived predictor-rank bound uses the smallest training-fold size and the
parameter `samples_per_predictor_rank`. A default of 10 was too restrictive for ordinary small
multivariate datasets and encouraged examples to bypass the rule by setting an explicit maximum
predictor rank. That gave the wrong impression that routine users should choose the search ceiling
themselves.

Both public selection interfaces already used five-fold regression cross-validation by default.

## Decision

`PiPLSRegression` and `PiPLSPathCV` both default to:

- `samples_per_predictor_rank=5`;
- `cv=5`.

The fold-safe upper bound remains

\begin{equation}
r_{\pi,\max}
=
\min\left[
p,
n_{\mathrm{train,min}},
\left\lceil
n_{\mathrm{train,min}} / 5
\right\rceil
\right].
\end{equation}

Values below 5 remain legal but emit `StatisticalSupportWarning` in rule-based modes. An explicit
integer predictor rank or path maximum remains available for a deliberate scientific protocol,
but ordinary examples and the Pulp smoke check do not set one.

## Consequences

- Basic estimator calls use the same rank-support and fold-count defaults.
- Small real-data examples demonstrate the rule-derived search rather than a hand-selected ceiling.
- The rank rule remains fold-local and leakage-safe.
- Benchmarks may still set a different `samples_per_predictor_rank` explicitly when that value is
  part of the benchmark's controlled scientific setup.
