# Decision 0013: final pre-D2 scikit-learn cleanup boundary

## Status

Accepted.

## Decision

Before advanced CV work, the public API completes a compact compatibility cleanup:

- `PiPLSRegression.inverse_transform` reconstructs predictors and responses through the fitted
  least-squares loadings and restores original units.
- `decomposition_` owns the canonical read-only Pi-PLS factorization arrays; Decision 0040 later
  removes the duplicate direct symbolic and diagnostic aliases.
- Cross-validated interfaces accept `cv=None` for standard five-fold regression CV and
  `scoring=None` for estimator scoring, expose `scorer_`, and include standard timing diagnostics.
- `PiPLSSearchCV` supports a direct `PiPLSRegression` or a `Pipeline` whose final step is
  `PiPLSRegression`. Arbitrary nested meta-estimators are rejected until explicitly supported.
- Delegated prediction and transformation methods are conditionally exposed when refitting is
  enabled and the selected estimator supports them.
- CI checks both the declared minimum scikit-learn series and the current dependency resolution.

## Consequences

The supported surface is narrower but reliable. Metadata routing and broader composite-estimator
support remain D2 or later work.
