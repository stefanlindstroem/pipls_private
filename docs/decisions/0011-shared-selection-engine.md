# Decision 0011: shared private selection engine

## Status

Accepted and implemented.

## Decision

`PiPLSRegression` remains the public estimator for fitting one fixed
`(n_components, predictor_rank)` model. `PiPLSPathCV` remains a meta-estimator that
searches and refits complete estimators or pipelines. Neither public class wraps the
other.

The reusable implementation boundary is private:

- `src/pipls/_cv_engine.py` owns fold-by-fold candidate evaluation, estimator cloning,
  parameter injection, scoring, response-standardized MSE, caching, parallel batches,
  and optional fitted-SVD diagnostics;
- `src/pipls/model_selection.py` owns exhaustive and adaptive one-dimensional
  predictor-rank search orchestration;
- `PiPLSRegression` uses those components for a fixed-`n_components` rank search;
- `PiPLSPathCV` uses the same components for each row of the triangular path.

A one-row `PiPLSPathCV` search and the corresponding `PiPLSRegression` rank search
must produce identical evaluated ranks, split-derived scores, selected rank, and
adaptive search history when supplied the same estimator settings and CV splits.

## Consequences

- fixes to fold-local preprocessing, scoring, caching, tie handling, or adaptive rank
  refinement have one implementation site;
- `PiPLSRegression` remains lightweight for fixed-rank fitting and does not acquire
  path-search metadata or meta-estimator overhead;
- `PiPLSPathCV` continues to support complete pipelines without creating a circular
  public dependency;
- future D2 out-of-fold and split-protocol work should build on the same private
  evaluation boundary rather than adding another fold loop.
