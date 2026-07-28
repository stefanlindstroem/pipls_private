# Decision 0115: normalize path-wide and selected-row result state

## Status

Accepted and implemented.

## Context

`PiPLSComponentPath` stored `predictor_rank_policy` and `n_splits` as row-aligned arrays even though
one fitted search uses one predictor-rank policy and one materialized cross-validation scheme for
every component count. The search therefore created those arrays with `np.full(...)`, and the result
validated hypothetical row-to-row variation that package code could never produce.

`PiPLSPredictorRankProfile` stored a complete `PiPLSComponentResult` in addition to the aligned
candidate arrays. Its constructor then located that selected rank in the arrays and revalidated that
all copied scalar scores matched the selected row. The profile already contains every quantity
needed to recover the selected result, except for the shared predictor-rank policy.

The public read surface is useful and should remain available. The package is unreleased, so the
redundant constructor state can be removed without a compatibility alias or transitional signature.

## Decision

`PiPLSComponentPath` stores these row-aligned read-only arrays:

- `n_components`;
- `predictor_rank`;
- `mean_test_score`;
- `cv_mse_mean`;
- `cv_mse_fold_sd`.

It stores `predictor_rank_policy` and `n_splits` once as validated Python scalars. Every
`PiPLSComponentResult` returned by path lookup or recommendation methods receives those shared
scalar values.

`PiPLSPredictorRankProfile` stores the aligned candidate arrays together with scalar
`n_components`, `predictor_rank_policy`, and `n_splits`. It no longer accepts or serializes a
`selected` result. The existing public `selected` attribute becomes a derived property that:

1. maximizes the stored `mean_test_score` values;
2. uses the same reference-anchored tolerant tie mask as fitted search selection;
3. returns the lowest predictor rank among tied rows, which is the first row because profile ranks
   are strictly ascending;
4. constructs a validated `PiPLSComponentResult` from that row and the shared scalar metadata.

This decision supersedes the array requirement for `predictor_rank_policy` and `n_splits` in
Decision 0066, the independently stored `selected` field in Decision 0072, and the corresponding
direct-construction consistency checks in Decision 0093. It does not change the fitted search,
candidate evaluation, score tolerance, tie-breaking, component recommendations, or public read
names.

## Consequences

- Path-wide facts are represented once rather than repeated for every component row.
- Predictor-rank profiles cannot contain a selected result inconsistent with their candidate arrays.
- `path.predictor_rank_policy` is a string and `path.n_splits` is an integer rather than arrays.
- `profile.selected` remains available but is recalculated from immutable profile state on access.
- Pickle reconstruction contains only independent arrays and shared scalar metadata.
- Historical hypothetical paths with mixed policies or split counts are no longer constructible.
