# Decision 0140: search-owned path selection

## Status

Accepted and implemented.

## Context

Decision 0137 separated path evaluation, final full-data fitting, and selection-conditioned OOF
reporting. A fitted `PiPLSSearchCV` owns candidate evidence, while `refit()` and
`validation_report()` resolve one stored path row through the same private selection machinery.

The public inspection surface still splits selection semantics across two objects. Users inspect
aligned evidence through `search.component_path_`, but retrieve a chosen row through methods on the
path object:

```python
selected = search.component_path_.for_n_components(3)
selected = search.component_path_.minimum_cv_mse_result()
selected = search.component_path_.one_standard_error_result()
```

This makes the path object both a numerical result container and a selection service. It also makes
introductory workflows resemble the former manual parameter-transfer design, even though
`search.refit()` now owns final estimator construction. Automatic named-rule inspection is less
discoverable than fitting because the corresponding lookup remains on `component_path_`.

Decision 0139 has already added the rendered Pulp quick start and placed it first in served
navigation. Its final documentation-reframing patch should target the final selection vocabulary
rather than migrate tutorials and references twice.

The package remains unreleased at version `0.0.0`, so selection ownership can move directly without
aliases, deprecation warnings, or compatibility methods.

## Decision

Make `PiPLSSearchCV` the sole public owner of path-selection semantics.

### Public operation

Add the post-fit method:

```python
def select(
    self,
    *,
    rule: SelectionRule | None = None,
    n_components: int | None = None,
) -> PiPLSComponentResult:
    ...
```

Exactly one of `rule` and `n_components` is required. Supported rules are:

```text
best_score
minimum_cv_mse
one_standard_error
```

A component-count selection returns the complete stored row $(h,r_\pi^*(h))$, including the
conditionally selected predictor rank. A rule selection returns the same immutable row that
`refit()` and `validation_report()` would resolve for that rule.

`select()`:

- requires a fitted search;
- returns an immutable `PiPLSComponentResult`;
- performs no fitting, prediction, candidate rescoring, or split materialization;
- does not mutate the search or attach selected state;
- does not retain supplied data or a returned result;
- uses the same exact stored-value and tie behavior as the current selection machinery.

Rename the private rule type from `RefitRule` to `SelectionRule`, because the vocabulary is shared by
selection inspection, final refitting, and validation reporting.

### Shared resolver

`select()`, `refit()`, and `validation_report()` use one private search-owned resolver.
`predictor_rank_profile(n_components=...)` uses the same internal component-row lookup when composing
its selected result. The implementation must not route public selection through methods that are
scheduled for removal from `PiPLSComponentPath`.

The numerical contracts remain unchanged:

- `"best_score"` resolves the global configured-score optimum;
- `"minimum_cv_mse"` resolves the first exact stored CV-MSE minimum;
- `"one_standard_error"` uses the minimum row's standard error and returns the smallest stored
  eligible component count;
- exact stored comparisons use no additional floating-point tolerance;
- the 1-SE rule requires a defined standard error from at least two validation splits.

### Path-object boundary

`PiPLSComponentPath` becomes an immutable aligned numerical inspection object. Its public role is to
expose:

```text
n_components
predictor_rank
mean_test_score
cv_mse_mean
cv_mse_fold_sd
cv_mse_standard_error
n_splits
predictor_rank_policy
```

Remove these public methods after all maintained consumers have migrated:

```text
for_n_components()
minimum_cv_mse_result()
one_standard_error_result()
```

A private indexed-row constructor may remain for search internals and result composition. It is not
a public selection surface.

### Migration sequence

Implement the transition in four reviewable patches:

1. establish this decision and the guide-layer target;
2. implement `PiPLSSearchCV.select()`, consolidate the private resolver, and temporarily retain the
   path-level methods while parity tests protect unchanged numerical behavior;
3. migrate maintained examples, tutorial renderers, tests, and living documentation to
   `search.select(...)`;
4. remove the three path-level methods, relocate their numerical contracts to search-selection
   tests, close the transition, and run active-surface audits.

Decision 0139 Patch 3 is paused after its first two completed patches. It resumes after this
transition so the landing page, tutorials, and path reference are reframed once around the final
selection API.

This decision refines Decisions 0066 and 0137 where public selection responsibility overlaps
with `PiPLSComponentPath`. The retired path-recommendation and constructor-selection records are
covered by the Decision 0147 retirement map.

## Implementation status

All four patches are implemented. `PiPLSSearchCV.select()` is the sole public selected-row lookup,
the shared rule type is `SelectionRule`, and `select()`, `refit()`, `validation_report()`, and
predictor-rank profile composition use search-owned helpers. Maintained examples, tutorial
renderers, tests, and living user documentation use `search.select(...)`. `PiPLSComponentPath`
contains only aligned numerical evidence, path-wide metadata, derived standard errors, immutable
serialization behavior, and the private indexed-row constructor used by search internals.

## Consequences

- `component_path_` has one clear role: plotting and numerical inspection.
- `search.select(...)`, `search.refit(...)`, and `search.validation_report(...)` share one selection
  vocabulary and one resolver.
- Users can inspect an automatic named-rule decision without fitting a model.
- Tutorials no longer teach path-level scalar lookup as a prerequisite for final fitting.
- Selection numerics and tie behavior do not change.
- No compatibility aliases or deprecated path methods remain.
