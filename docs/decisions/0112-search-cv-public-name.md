# Decision 0112: search-CV public name

## Status

Accepted.

## Context

The package's cross-validated meta-estimator was named `PiPLSPathCV`. The class does produce an
immutable component path, but its primary public responsibility is to search admissible
`(n_components, predictor_rank)` candidates, select one result under a declared rule, and
optionally refit that model. For a reader who does not already know the Pi-PLS path terminology,
`PathCV` does not state that responsibility directly.

Scikit-learn uses the `SearchCV` suffix for meta-estimators that evaluate parameter candidates by
cross-validation. Pi-PLS has a structured triangular candidate domain rather than a rectangular
parameter grid, but `SearchCV` does not imply exhaustive or rectangular evaluation. The existing
`search_method` parameter continues to distinguish adaptive and exhaustive search.

The package is unreleased at version `0.0.0`, so the former name has no release-compatibility
obligation.

## Decision

1. Rename the public meta-estimator from `PiPLSPathCV` to `PiPLSSearchCV`.
2. Export only `PiPLSSearchCV` from `pipls`; do not retain a compatibility alias, deprecation shim,
   or legacy import module.
3. Move the implementation module from `src/pipls/path.py` to `src/pipls/search.py` so the source
   layout follows the public responsibility.
4. Retain `PiPLSComponentPath`, `PiPLSComponentResult`, `component_path_`,
   `predictor_rank_profile()`, and path-analysis terminology where they describe stored path
   results rather than the search object itself.
5. Retain all constructor parameters, fitted attributes, numerical behavior, selection rules,
   refit behavior, and deterministic contracts unchanged.
6. Update source annotations, exception messages, examples, benchmarks, generated tutorial
   sources, tests, public documentation, historical decision references, guide-layer records,
   distribution checks, and changelog in the same patch.

## Consequences

The fixed and selection responsibilities are expressed directly:

```python
model = PiPLSRegression(
    n_components=3,
    predictor_rank=10,
).fit(X, Y)

search = PiPLSSearchCV().fit(X, Y)
path = search.component_path_
```

New users can infer that `PiPLSSearchCV` performs cross-validated model search without first
learning the package's component-path terminology. The path result remains explicit and retains
its established mathematical and inspection meaning. Code written against the unreleased former
name must adopt the new import directly.
