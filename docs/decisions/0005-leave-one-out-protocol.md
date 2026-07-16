# Decision: 0005 leave-one-out and advanced split protocol

## Status

Accepted and implemented in Phase D2.

## Decision

Leave-one-out is represented by the ordinary scikit-learn `LeaveOneOut()` splitter. The numerical
core has no LOO branch or Boolean mode. The fold-safe rank bound uses the smallest training fold;
for LOO this is `n - 1` in both the algebraic cap and samples-per-rank term.

The default performance criterion is response-standardized MSE using response scales estimated
from each matching training fold with `ddof=1`. Ordinary R2 scoring is rejected for singleton
validation folds. A pooled R2 may be calculated from row-ordered OOF predictions only as a
secondary diagnostic.

Both public interfaces accept `return_oof_predictions=True`. OOF predictions are generated after
selection by refitting the selected fixed parameterization on every materialized training fold.
Rows are retained in original order; repeated predictions are averaged; per-row validation counts
are exposed; uncovered rows contain NaN.

`PiPLSValidationReport` labels estimates as either `selection-conditioned` or `fixed-parameter`.
The path result and internally selected regression result are selection-conditioned because the
same CV results choose and report the parameterization. They must not be described as unbiased
external-test or nested-CV estimates.

Group metadata is accepted explicitly by both public `fit` methods and participates in
scikit-learn metadata routing. Repeated, predefined, grouped, temporal, and LOO splitters are
covered by public tests.
