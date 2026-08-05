# Decision 0066: immutable component-path API

## Status

Accepted and implemented. Decisions 0140, 0145, and 0146 refine selection ownership, public names,
and CV-MSE tolerance provenance.

## Context

A fitted search needs a concise component-path result with one conditionally selected predictor
rank for every evaluated component count. The result must remain stable after fitting, safe to
serialize, and independent of the large candidate-level `cv_results_` mapping.

Selection is a separate concern. A path is aligned numerical evidence; the fitted search owns named
selection rules, manual component-count lookup, full-data refitting, and OOF reporting.

## Decision

`PiPLSSearchCV.component_path_` is an immutable `PiPLSComponentPath` containing aligned read-only
arrays:

```text
n_components
predictor_rank
mean_test_score
cv_mse_mean
cv_mse_std
```

and path-wide scalar provenance:

```text
predictor_rank_policy
n_splits
```

Each row contains the predictor rank selected conditionally for that component count under the
configured scorer and tie-breaking rule. Component counts are unique and strictly increasing.
`cv_mse_std` is population SD across the materialized validation splits.

The path exposes no public selected-row lookup or recommendation methods. Use:

```python
selection = search.select(rule="minimum_cv_mse")
selection = search.select(rule="best_score")
selection = search.select(n_components=3)
```

Each operation returns an immutable `PiPLSSelection`. A selection contains one evaluated rank pair,
its score and CV-MSE summaries, the path policy, split count, and optional rule provenance. A
minimum-CV-MSE selection additionally retains the exact unruled minimum row, resolved relative and
absolute tolerances, and a derived effective threshold.

All public path and selection records defensively copy arrays, normalize scalars, validate direct
construction, remain read-only, and reconstruct through the same validation when unpickled.

`cv_results_` remains the candidate-level evidence table. It is not duplicated inside the concise
path or scalar selections.

## Consequences

- Component-path evidence has one stable row per component count.
- Selection ownership is unambiguous and search-owned.
- No `best_*` fitted state, path recommendation method, compatibility alias, or mutable result view
  is required.
- User code can serialize, compare, and plot path arrays without retaining training data.
