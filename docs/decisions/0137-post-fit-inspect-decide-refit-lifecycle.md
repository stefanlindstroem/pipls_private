# Decision 0137: post-fit inspect-decide-refit lifecycle

## Status

Accepted. Implementation is authorized but not yet complete.

## Context

`PiPLSSearchCV` currently requires the final selection rule and refit policy to be declared before
`fit()`. That follows the general `GridSearchCV` pattern, but it works against a central Pi-PLS use
case: users often need to inspect the component path and one or more conditional predictor-rank
profiles before choosing the final model.

The current constructor-time lifecycle also couples path evaluation, selected-row bookkeeping,
optional out-of-fold (OOF) reporting, full-data refitting, and model delegation on one object. The
result is more state and more conditional behavior than a pre-release package needs.

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

When this transition is complete, remove the constructor parameters `selection_rule`, `refit`, and
`return_oof_predictions`; remove `selected_result_`, `selected_params_`, `validation_report_`,
`selected_estimator_`, `selected_pipls_`, and `refit_time_`; and remove search-level prediction,
transformation, scoring, inverse-transformation, and feature-name delegation. The fixed estimator
or returned pipeline owns fitted-model behavior.

Retain `cv_results_`, `component_path_`, `predictor_rank_profile()`, `best_index_`, `best_score_`,
`best_params_`, `best_n_components_`, `best_predictor_rank_`, `search_is_exhaustive_`, and
`scorer_` as search evidence. The `best_*` attributes continue to identify the global optimum under
the configured scorer; they do not imply a final model choice.

Implement the transition in reviewable stages: add `refit()`, add explicit validation reporting,
remove the old lifecycle, then migrate examples and public documentation. Until the old lifecycle
is removed, public user documentation continues to describe the implemented API rather than the
future target.

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
- The staged implementation may temporarily contain both lifecycles, but the completed version
  contains no legacy compatibility surface.
