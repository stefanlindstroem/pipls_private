# Decision 0114: derive prediction diagnostics from independent inputs

## Status

Accepted and implemented.

## Context

`PredictionDiagnostics` accepted ten constructor fields even though seven were completely determined
by `observed` and `predicted`: residuals, three standardized matrices, response centers, response
scales, and standardized RMSE. The constructor then recomputed every derived quantity solely to
check that the supplied fields were mutually consistent.

The derived attributes are useful for inspection and direct rendering and must remain available.
Their finite float64, response-orientation, standardization, and read-only contracts must also remain
unchanged.

## Decision

`PredictionDiagnostics` accepts only the independent constructor inputs:

- `observed`;
- `predicted`;
- `prediction_kind`.

Direct construction validates the two aligned finite response matrices and the provenance label,
then derives the residual, observed-response centers and sample scales, standardized observed and
predicted values, standardized residual, and response-wise standardized RMSE once. These derived
arrays remain stored as read-only public attributes.

Pickle reconstruction serializes the independent inputs and passes through the same constructor, so
all derived arrays are recalculated and revalidated after restoration. `prediction_diagnostics()`
continues normalizing one-dimensional response input to explicit two-dimensional orientation before
constructing the result.

The existing checked arithmetic remains authoritative. Constant observed-response columns and any
derived quantity that cannot be represented as finite float64 are rejected before an instance is
returned.

This decision supersedes the independent-constructor-field and relationship-validation requirement
for `PredictionDiagnostics` in Decision 0094. It does not change residual orientation, `ddof=1`
scaling, prediction-provenance labels, public derived attribute names, array shapes, or numerical
values.

## Consequences

- Inconsistent diagnostic state is impossible through public construction.
- Seven redundant constructor arguments and their pairwise consistency checks are removed.
- Existing inspection and plotting code continues reading the same derived attributes.
- Pickled records contain only the independent response matrices and prediction provenance.
- The package remains pre-release, so no compatibility constructor is retained.
