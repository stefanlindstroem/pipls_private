# Decision 0137: post-fit inspect-decide-refit lifecycle

## Status

Accepted and implemented. Decisions 0140, 0143, 0146, and 0148 refine selected-row ownership,
selection provenance, OOF reporting, hierarchical rank retention, and the final rule vocabulary.
Decision 0151 refines the evidence-to-refit order and lets one existing selection configure both OOF
reporting and final fitting.

## Context

A path search is evidence, not a final fitted model. Constructor-owned automatic refitting mixed
cross-validated path evaluation, scientific inspection, final full-data fitting, and validation
reporting in one mutable object. It also encouraged users to treat one automatically selected row as
the only meaningful output of a search.

## Decision

`PiPLSSearchCV.fit(X, y)` evaluates and stores the admissible path only. It materializes one split
set, retains no training matrices, and fits no final full-data estimator.

Post-fit operations are explicit:

```python
search = PiPLSSearchCV(cv=cv).fit(X, y)
path = search.component_path_

selection = search.select(rule="minimum_cv_mse")
profile = search.predictor_rank_profile(selection.n_components)
report = search.oof_report(X, y, selection=selection)
model = search.refit(X, y, selection=selection)
```

`select()` resolves one stored row without fitting. `refit()` either resolves a rule or component
count directly or consumes an existing compatible selection, clones the configured estimator or
pipeline, fits it on all supplied data, and attaches the exact immutable selection as
`model.selection_`. `oof_report()` accepts an existing compatible selection, refits its
fixed rank pair on every split materialized by the search, averages repeated validation predictions
per observation, and returns immutable ordered diagnostics.

The supported named rules are `best_score` and tolerance-based `minimum_cv_mse`. Manual selection
uses an evaluated `n_components` value and the predictor rank already selected conditionally for
that row. Users who need an exact manually specified `(n_components, predictor_rank)` pair fit
`PiPLSRegression` directly.

The search retains path evidence such as `cv_results_`, `component_path_`, `n_splits_`,
`max_predictor_rank_`, `search_is_exhaustive_`, and `scorer_`. It does not expose fitted `best_*`
state, delegate prediction to a hidden final estimator, or mutate when selection, refitting, or OOF
reporting succeeds.

## Consequences

- Path evaluation, row selection, final fitting, and OOF reporting have distinct operations and
  ownership.
- Selection can be inspected before deciding whether to fit a final model.
- Refit and OOF operations use the same stored pair and fitted search evidence.
- OOF reporting is selection-conditioned descriptive validation, not nested-CV or external-test
  performance.
- No constructor-time refit switch, selected fitted-model state, compatibility alias, or stored
  training matrix is required.
