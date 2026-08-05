# Decision 0140: search-owned path selection

## Status

Accepted and implemented. Decisions 0143, 0146, and 0148 define the final provenance,
component-count tolerances, and conditioned-path rule semantics.

## Context

The concise component path is aligned numerical evidence. Public row lookup and automatic
selection previously appeared in several path methods and search operations, which created
competing ownership and made it possible for selection semantics to drift.

## Decision

`PiPLSSearchCV.select()` is the sole public selected-row lookup:

```python
search.select(rule="best_score")
search.select(rule="minimum_cv_mse")
search.select(n_components=3)
```

Exactly one of `rule` and `n_components` is supplied. The method returns an immutable
`PiPLSSelection` and performs no fitting.

Rule semantics are:

- `best_score`: the maximum configured-score row on the predictor-rank-conditioned component
  path, with deterministic smaller-component and smaller-rank ordering for exact numerical ties;
- `minimum_cv_mse`: the smallest conditioned component-path row satisfying simultaneous relative
  and absolute tolerances around the exact minimum mean CV-MSE path row.

Manual component-count selection returns the stored row at that count with no rule provenance. The
predictor rank is the rank already selected conditionally for that component count.

`refit()` uses the same resolver and attaches the exact result to the returned model as
`selection_`. `oof_report()` accepts an existing selection and validates it against the same fitted
search evidence before reusing every materialized split.

`PiPLSComponentPath` exposes arrays only. It has no public `for_n_components()`, recommendation,
minimum, or threshold methods. `predictor_rank_profile()` remains a search method because it derives
candidate rows from `cv_results_`.

## Consequences

- One public operation owns scalar row lookup.
- Path evidence, rank-profile evidence, selection, fitting, and OOF reporting remain distinct.
- Named-rule provenance survives refitting, pickling, and OOF compatibility validation.
- No duplicated path-selection helpers or fitted `best_*` aliases are required.
