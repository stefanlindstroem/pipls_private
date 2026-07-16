# Decision 0010: pipeline-aware Pi-PLS path analysis

## Status

Accepted and implemented.

## Decision

`PiPLSPathCV` is the public two-parameter search layer. Its default `"auto"` mode
applies the established adaptive predictor-rank search independently for each component count. Its `"optimal"` mode evaluates the complete
admissible triangular surface.

The complete estimator or pipeline is cloned inside every training fold and candidate fit.
A unique nested `PiPLSRegression` is inferred, while deeper composites use an explicit
`pipls_param_prefix`.

The fold-safe default predictor-rank limit uses the smallest training-fold size and the
smallest predictor dimension reaching the Pi-PLS step. An explicit integer
`max_predictor_rank` bypasses the statistical samples-per-rank rule but not fold-safe
algebraic limits.

Global ties prefer smaller `n_components`, then smaller `predictor_rank`. Conditional ties
for one component count prefer smaller predictor rank. The selected complete estimator is
refitted on all supplied data when `refit=True`.

## Consequences

- arbitrary learned preprocessing remains inside the CV boundary;
- the exhaustive and adaptive vocabulary is consistent with `PiPLSRegression`;
- complete path diagnostics are available without nested automatic rank selection;
- ordered out-of-fold predictions and paper-specific LOO reporting remain Phase D2 work.
