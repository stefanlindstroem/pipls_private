# Decision 0093: validate immutable core public results

## Status

Accepted.

## Context

The public fixed-estimator and path layers return five frozen result records:
`PiPLSDecomposition`, `PiPLSComponentResult`, `PiPLSPredictorRankProfile`,
`PiPLSComponentPath`, and `PiPLSValidationReport`. Estimator-produced instances were mostly
well formed, but direct construction did not apply one consistent policy. Some records retained
aliased writable arrays, some normalized values and others did not, and several accepted invalid
scalar, score, or coverage states.

The classes are public types even though the generated reference presents them as returned records
rather than constructor-first APIs. Their invariants must therefore hold regardless of whether an
instance originates from a fitted estimator, a user, or pickle reconstruction.

## Decision

Apply one defensive validation contract to all five records.

1. Copy every stored NumPy array into its documented dtype and make it read-only.
2. Normalize accepted NumPy scalar integers, floats, and booleans to the corresponding Python
   scalar types. Reject booleans where an integer or real value is required.
3. Require positive component, predictor-rank, and split counts, with
   `n_components <= predictor_rank`.
4. Require finite scores and diagnostics. Response-standardized MSE values and fold standard
   deviations are nonnegative.
5. Require aligned one-dimensional path/profile arrays, strictly ascending unique component or
   predictor-rank indices, supported predictor-rank policies, and an exact match between a
   predictor-rank profile's selected record and its aligned row.
6. Require decomposition arrays to be finite, nonempty, and component-aligned; dilation is
   nonnegative; numerical rank is at least the retained component count; and solver provenance
   agrees with whether the reported numerical rank is exact.
7. Require validation-report OOF counts to be nonnegative integer values aligned with prediction
   rows. Covered rows contain only finite predictions; uncovered rows contain only NaN. OOF counts
   and pooled OOF $R^2$ cannot exist without OOF predictions.
8. Reconstruct every record through its validating constructor during unpickling so read-only and
   scalar invariants survive serialization.
9. Keep constructor signatures suppressed in generated reference pages. The records remain
   returned-first interfaces, but direct construction is supported and validated.

No compatibility layer is required because the package has not been released. Invalid states that
were previously constructible are rejected immediately.

## Consequences

Estimator and path outputs retain their existing values and field names, while public records have
one predictable immutability and validation boundary. Tests can construct records directly without
creating weaker states than the package itself returns.

The next pre-release hardening increment applies the same policy to inspection results and hardens
inspection calculations against nonfinite derived values.
