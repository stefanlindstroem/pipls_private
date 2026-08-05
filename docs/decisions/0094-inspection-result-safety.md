# Decision 0094: validate inspection results and derived numerical values

## Status

Accepted.

## Context

The public inspection layer returns five frozen records: `LatentStructure`,
`BiplotCoordinates`, `PiPLSDisplayFactors`, `ObservationDiagnostics`, and
`PredictionDiagnostics`. The helper functions normally copied their arrays, but direct construction
could retain writable aliases or inconsistent fields. Several calculations also used direct sums of
squares, norms, means, covariance products, and factor products. Finite inputs near the float64
range could therefore emit runtime warnings and return `inf` or `nan` inspection values.

Inspection results are public numerical records. They must enforce the same defensive boundary as
the core estimator and path records, and finite inputs must not silently produce nonfinite public
quantities.

## Decision

1. Direct construction of every inspection record copies arrays to float64 or platform integers,
   makes them read-only, validates documented dimensions and aligned shapes, and rejects nonfinite
   fields. Nonnegative or positive quantities retain those constraints.
2. `PiPLSDisplayFactors` validates that `weighted_response_directions` equals
   `response_directions * dilation`. `PredictionDiagnostics` validates its residual,
   standardization, center, scale, RMSE, and provenance relationships. Weighted response
   directions and prediction-diagnostic dependent arrays are derived from independent inputs while
   retaining the same finite-value boundary.
3. Pickle reconstruction passes through the same validating constructors.
4. Biplot norms use max-scaled Euclidean calculations, and the balancing factor is formed as a
   quotient of square roots rather than by first forming a potentially overflowing norm ratio.
5. Prediction centers, sample scales, and standardized RMSE use scaled calculations. Residual,
   centering, standardization, and factor products are checked immediately for finite float64
   representability.
6. Observation score covariance is formed after one common finite scaling of the centered training
   scores. This preserves the Moore--Penrose score distance while avoiding overflow in the covariance
   product. X-reconstruction residuals use scaled row-wise squared norms.
7. If a mathematically requested inspection quantity cannot be represented as finite float64, the
   helper raises `ValueError` naming that quantity. Expected floating-point warnings are suppressed
   only around the checked operation; nonfinite results never escape.
8. These safeguards do not add thresholds, probability limits, outlier labels, uncertainty claims,
   or rendering behavior.

No compatibility layer is required because the package has not been released.

## Consequences

Ordinary fitted-model and prediction results retain their existing values, shapes, and field names.
Extreme finite inputs succeed when the requested result is representable and fail explicitly when it
is not. Public inspection records now have the same direct-construction, immutability, and pickle
boundary as the core public results.

The final pre-release hardening increment closes recursively frozen dataset metadata by rejecting
object-dtype arrays.
