# Decision 0011: shared private selection engine

## Status

Refined and implemented by Decision 0039. `PiPLSPathCV` is the sole public selection interface;
the retained private fold and adaptive-search helpers are path-owned and contain no duplicate
fixed-estimator selection implementation.

## Decision

`PiPLSRegression` remains the public estimator for fitting one fixed
`(n_components, predictor_rank)` model. `PiPLSPathCV` remains a meta-estimator that
searches and refits complete estimators or pipelines. Neither public class wraps the
other.

The remaining reusable implementation boundary is private and path-owned:

- `src/pipls/_cv_engine.py` owns fold-by-fold candidate evaluation, estimator cloning,
  parameter injection, scoring, response-standardized MSE, caching, parallel batches,
  and optional fitted-SVD diagnostics;
- `src/pipls/model_selection.py` owns exhaustive and adaptive one-dimensional
  predictor-rank search orchestration;
- `PiPLSPathCV` uses those components for each row of the triangular path.

`PiPLSRegression` no longer uses the selection engine. It fits one explicit fixed pair.

## Consequences

- fixes to fold-local preprocessing, scoring, caching, tie handling, or adaptive rank
  refinement have one implementation site;
- `PiPLSRegression` remains lightweight for fixed-rank fitting and owns no selection machinery;
- `PiPLSPathCV` continues to support complete pipelines without creating a circular
  public dependency;
- future D2 out-of-fold and split-protocol work should build on the same private
  evaluation boundary rather than adding another fold loop.
