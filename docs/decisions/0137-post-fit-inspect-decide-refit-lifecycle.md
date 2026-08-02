# Decision 0137: post-fit inspect-decide-refit lifecycle

## Status

Accepted and implemented.

## Context

Before this decision, `PiPLSSearchCV` required the final selection rule and refit policy to be
declared before `fit()`. That followed the general `GridSearchCV` pattern, but worked against a
central Pi-PLS use case: users often need to inspect the component path and one or more conditional
predictor-rank profiles before choosing the final model.

That constructor-time lifecycle also coupled path evaluation, selected-row bookkeeping, optional
out-of-fold (OOF) reporting, full-data refitting, and model delegation on one object. The result was
more state and more conditional behavior than a pre-release package needs.

The package remains unreleased at version `0.0.0`. No compatibility layer is required for the
current constructor controls or fitted selected-model attributes.

## Decision

Adopt an inspect-decide-refit lifecycle.

A fitted `PiPLSSearchCV` is a path evaluator and evidence object. `fit(X, Y)` evaluates candidates,
materializes the validation splits needed for reproducible follow-up reporting, and exposes the
complete candidate table plus immutable inspection views:

```python
search = PiPLSSearchCV(cv=cv).fit(X, Y)
path = search.component_path_
profile = search.predictor_rank_profile(n_components=4)
```

Final model fitting is an explicit post-fit operation:

```python
model = search.refit(X, Y, rule="one_standard_error")
model = search.refit(X, Y, n_components=4)
```

`refit()` accepts exactly one of `rule` and `n_components`. Supported rules are
`"best_score"`, `"minimum_cv_mse"`, and `"one_standard_error"`. A manual component count uses the
predictor rank already selected conditionally for that component-path row. A caller who needs an
exact manually specified `(n_components, predictor_rank)` pair uses `PiPLSRegression` directly.

`refit()` clones the configured direct estimator or terminal-Pi-PLS pipeline, sets the selected
Pi-PLS rank pair, fits the clone on the supplied full data, and returns that fitted estimator. It
does not mutate the search object, store the supplied `X` or `Y`, or attach the returned estimator
to search state.

Selection-conditioned OOF reporting becomes a separate explicit post-fit operation:

```python
report = search.validation_report(X, Y, rule="one_standard_error")
report = search.validation_report(X, Y, n_components=4)
```

`validation_report()` uses the same selected-row resolver as `refit()`, reuses the exact validation
splits materialized by `fit()`, returns an immutable `PiPLSValidationReport`, and does not perform a
full-data refit or mutate the search object.

Remove the former constructor-time selection, refit, and OOF controls; remove selected-row and
selected-model fitted state from the search; and remove search-level prediction, transformation,
scoring, inverse-transformation, and feature-name delegation. The fixed estimator or returned
pipeline owns fitted-model behavior.

Retain `cv_results_`, `component_path_`, `predictor_rank_profile()`, `best_index_`, `best_score_`,
`best_params_`, `best_n_components_`, `best_predictor_rank_`, `search_is_exhaustive_`, and
`scorer_` as search evidence. The `best_*` attributes continue to identify the global optimum under
the configured scorer; they do not imply a final model choice.

Implement the transition in reviewable stages: add `refit()`, add explicit validation reporting,
remove the remaining old lifecycle, then migrate examples and final presentation. Public user
documentation describes the implemented stage rather than claiming unfinished behavior.

The original staging expected the constructor boolean for automatic refitting and the post-fit
`refit()` method to coexist temporarily. That is not a valid scikit-learn estimator surface:
constructor parameters are stored as same-named instance attributes, so the boolean would shadow
the method. Post-fit `refit()` therefore arrived together with removal of constructor refitting,
selected fitted-model state, and search-level fitted-model delegation. Explicit
`validation_report()` then replaced constructor-owned OOF reporting and reuses defensive read-only
copies of the exact materialized split indices. The final cleanup removed all remaining
constructor-selected report state without aliases or deprecation machinery.

Maintained examples and tutorial renderers use `search.refit(...)` for the final full-data model
rather than manually constructing a second estimator from the selected rank pair. Scalar result
lookups remain available when a workflow needs the stored row for plotting or reporting.

Do not add aliases, deprecation warnings, ignored constructor arguments, fallback attributes, or
serialization migrations for the removed pre-release surface.

This decision supersedes Decisions 0005, 0066, 0087, 0102, 0111, 0116, and 0129 only where they
require constructor-time final selection, selected-model state on the search object, or
constructor-owned OOF reporting. Their splitter, immutable-result, scoring, and numerical contracts
otherwise remain in force.

## Consequences

- Path evaluation, scientific inspection, final model fitting, and OOF reporting have distinct
  operations and result ownership.
- Manual inspection can precede either a named automatic rule or a chosen component count.
- The shortest automatic workflow remains compact:

  ```python
  model = PiPLSSearchCV(cv=cv).fit(X, Y).refit(
      X,
      Y,
      rule="one_standard_error",
  )
  ```

- Retaining the search variable preserves access to component paths, predictor-rank profiles, and
  candidate diagnostics; discarding it intentionally retains only the fitted final model.
- The returned model remains an ordinary `PiPLSRegression` or user-supplied pipeline rather than a
  search object with delegated fitted-model methods.
- Search objects do not retain potentially large or sensitive training matrices.
- The completed public search surface contains no compatibility aliases, ignored arguments, or
  constructor-selected report state.
