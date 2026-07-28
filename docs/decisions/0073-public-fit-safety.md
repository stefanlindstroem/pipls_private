# Decision 0073: public fit-state and finite-output safety

## Status

Accepted and implemented.

## Context

The fixed numerical core is private and receives centered or standardized arrays from package-owned
callers. It is not useful to add extensive defensive policy to every private matrix operation.
The public estimators, however, must not report a successful fit with nonfinite fitted values or
retain an earlier fitted model after a later fit fails.

Two concrete boundary failures required correction. Very small but finite columns could underflow
when sample standard deviations were calculated, allowing a successful fit with nonfinite
original-unit coefficients. A failed second fit could also leave the previous fitted attributes
available even after constructor parameters or input dimensions had changed. Read-only or
memory-overlapping arrays with `copy=False` introduced further public surprises.

## Decision

Harden the public estimator boundary while keeping the private numerical core focused:

- `PiPLSRegression.fit()` and `PiPLSSearchCV.fit()` are transactional. They remove existing fitted
  state before fitting and remove partial state after any exception.
- Ordinary means and sample standard deviations remain unchanged for normal data. Range-safe
  calculations are used only when the ordinary calculation overflows or when a nonconstant finite
  column underflows to a zero scale.
- `copy=False` remains permission to reuse writable storage, not a requirement that callers provide
  writable arrays. Read-only inputs are copied, and one block is copied when predictor and response
  arrays share memory.
- A public fixed fit rejects input magnitudes that can overflow the core cross-product and rejects
  nonfinite factorization, coefficient, intercept, score, or loading results rather than publishing
  them.
- Public prediction, transformation, and inverse reconstruction reject nonfinite outputs.
- A low-level `numpy.linalg.LinAlgError` arising during a public fixed fit is translated to a
  concise fit-level `ValueError` while preserving the original exception as its cause.
- Response-standardized MSE rejects a result that cannot be represented as a finite float64 value.

The private `fit_pipls_core()` contract remains unchanged: package-owned callers supply finite,
preprocessed matrices and retain responsibility for interpreting failures at the public boundary.

## Consequences

- A successful public fit has finite exposed numerical state.
- A failed initial fit or refit leaves the estimator unfitted according to scikit-learn checks.
- Ordinary-scale reference results and preprocessing values remain unchanged.
- Subnormal finite columns are not misclassified as constant solely because variance arithmetic
  underflows.
- Models whose original-unit coefficients are outside float64 range fail explicitly instead of
  returning `NaN` or infinity.
- The implementation adds a small public-boundary layer rather than pervasive private-core checks.
- Existing ordinary-scale numerical tolerances remain unchanged; the subnormal regression boundary
  test uses relative tolerance $10^{-12}$.
