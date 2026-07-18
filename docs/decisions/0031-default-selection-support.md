# Decision 0031: default selection support and fold count

## Status

Accepted and implemented for `PiPLSPathCV`. Decision 0039 removes the rank-bound parameters and
CV defaults from `PiPLSRegression`; the sample-count convention remains refined by Decision 0032.

## Context

The public rule-derived predictor-rank bound uses the parameter
`samples_per_predictor_rank`. A default of 10 was too restrictive for ordinary small
multivariate datasets and encouraged examples to bypass the rule by setting an explicit maximum
predictor rank. That gave the wrong impression that routine users should choose the search ceiling
themselves.

The package selection interface uses five-fold regression cross-validation by default.

## Decision

`PiPLSPathCV` defaults to:

- `samples_per_predictor_rank=5`;
- `cv=5`.

The defaults remain unchanged. Decision 0032 subsequently refined the upper bound to

\begin{equation}
r_{\pi,\max}
=
\min\left[
p,
n_{\mathrm{train,min}}-1,
\left\lceil n / 5\right\rceil
\right].
\end{equation}

Values below 5 remain legal but emit `StatisticalSupportWarning` in rule-based modes. An explicit
integer predictor rank or path maximum remains available for a deliberate scientific protocol,
but ordinary examples do not set one.

## Consequences

- Basic path calls use the documented rank-support and fold-count defaults.
- Small real-data examples demonstrate the rule-derived search rather than a hand-selected ceiling.
- The support term describes the full-data refit, while all learned quantities remain fold-local
  and leakage-safe during CV.
- Benchmarks may still set a different `samples_per_predictor_rank` explicitly when that value is
  part of the benchmark's controlled scientific setup.
