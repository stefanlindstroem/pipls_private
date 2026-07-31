# Decision 0132: predicate-style search exhaustiveness name

## Status

Accepted and implemented.

## Context

`PiPLSSearchCV` reports whether the completed search evaluated every admissible
`(n_components, predictor_rank)` pair. The pre-release fitted attribute
`path_search_exhaustive_` encoded that meaning, but its noun-heavy word order did not read naturally
as a boolean predicate.

The package remains unreleased at version `0.0.0`, so no published compatibility commitment
requires retaining the former spelling.

## Decision

Rename the fitted boolean attribute `path_search_exhaustive_` to `search_is_exhaustive_`.

The attribute remains true exactly when the number of evaluated admissible pairs equals the total
number of admissible pairs. It therefore reports the observed coverage of the completed search,
not merely the configured `search_method`. An optimal search is exhaustive, while an adaptive
search may also be exhaustive if it happens to evaluate the complete admissible set.

Use the predicate-style name throughout implementation, tests, public documentation, generated
source documentation, and active guide-layer contracts. Do not add an alias, fallback lookup, or
serialization migration for the former unreleased name.

Historical decision records remain unchanged. This decision supersedes Decisions 0087 and 0105
only where they require the former fitted-attribute spelling. Their fitted-surface and documentation
alignment contracts otherwise remain in force.

## Consequences

- The fitted boolean reads directly as a predicate.
- Search coverage semantics and all candidate evaluations remain unchanged.
- Existing pre-release code using `path_search_exhaustive_` must use `search_is_exhaustive_`.
- No duplicate compatibility attribute enlarges the fitted public surface.
