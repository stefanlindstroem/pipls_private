# Decision 0025: model-internal standardization boundary

## Status

Accepted while clarifying the package-product roadmap after Decision 0024.

## Context

Decision 0024 deferred the design of future standardization pipelines and block-scaling
functionality. That wording could be misread as deferring the centering and scaling already required
when fitting `PiPLSRegression`.

The current estimator already mirrors the relevant `PLSRegression` behavior: model fitting includes
centering of both predictor and response blocks and, when `scale=True`, division by training-sample
standard deviations. Rank and path selection clone and fit the complete estimator independently in
each training fold, then refit the selected model on all training observations.

The project owner clarified that future block scaling is an alternative model-standardization rule
inside the same leakage-safe fitting and validation boundary. It must not become a transform fitted
once to the complete dataset before cross-validation.

## Decision

- Treat centering and optional scaling as current, integral `PiPLSRegression` behavior, not as a
  deferred preprocessing feature.
- Every estimator fit learns `x_mean_` and `y_mean_` from the observations supplied to that fit.
- With `scale=True`, every fit also learns safe `x_scale_` and `y_scale_` vectors from those
  observations using the accepted sample-standard-deviation convention. With `scale=False`,
  centering remains active and the scale vectors are ones.
- During internal rank or path selection, clone and fit the complete estimator inside each training
  fold. Validation observations must not influence centering or scaling statistics.
- After selection, refit the chosen estimator on the complete training set supplied to `fit()`,
  including fresh estimation of its means and scales.
- Apply the stored training statistics at prediction time and return responses in their original
  units.
- Keep the fixed numerical core independent from preprocessing: it continues to receive centered or
  centered-and-scaled matrices from the estimator layer.
- Defer only the public API, naming, placement, and block semantics of future block-aware
  standardization. It may eventually be estimator-owned or represented in a supported model
  pipeline; that design is not decided. In either case, the complete candidate must fit it within
  each training fold and the final full-training refit.
- Reject workflows that fit learned scaling once to the complete dataset before cross-validation;
  that is leakage and is not an accepted package pattern.

## Consequences

- Existing runtime behavior remains unchanged.
- Documentation and future development must distinguish current estimator standardization from
  deferred block-aware variants.
- The package may later extend model standardization without changing the leakage-safe ownership
  boundary.
- No block-scaling class, parameter, composition rule, or implementation schedule is created by
  this decision.
