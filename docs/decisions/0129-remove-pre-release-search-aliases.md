# Decision 0129: remove pre-release search compatibility aliases

## Status

Accepted and implemented.

## Context

`PiPLSSearchCV` originally exposed `best_estimator_` and `best_pipls_` in addition to the standard
candidate-level `best_*` results. Decision 0111 later introduced explicit final selection rules and
the canonical `selected_estimator_` and `selected_pipls_` attributes. The fitted-estimator aliases
then existed only for the default best-score rule, duplicated the selected-model state, and made the
public fitted surface depend on which selection rule produced the same kind of final model.

The package remains unreleased at version `0.0.0`. No published compatibility commitment requires
retaining these duplicate names.

## Decision

Remove `PiPLSSearchCV.best_estimator_` and `PiPLSSearchCV.best_pipls_`.

When `refit=True`, `selected_estimator_` contains the one estimator refitted on all supplied data and
`selected_pipls_` contains its fitted terminal `PiPLSRegression`, regardless of whether
`selection_rule` is `"best_score"` or `"one_standard_error"`. Delegated prediction,
transformation, inverse transformation, scoring, output naming, and feature-name methods continue
to use `selected_estimator_`.

The candidate-level attributes `best_index_`, `best_score_`, `best_params_`,
`best_n_components_`, and `best_predictor_rank_` remain unchanged and continue to describe the
global configured-score optimum. `selected_result_` and `selected_params_` continue to describe the
declared final row.

Remove tests and active documentation that preserve the fitted-estimator aliases. Under Decision
0126, also remove executable tombstones for the former unreleased `PiPLSPathCV` name; historical
records continue to document that rename.

This decision supersedes Decision 0012 and Decision 0111 only where they require
`best_estimator_` or `best_pipls_`. Their remaining scikit-learn alignment, selection-result, and
refit-delegation contracts remain in force.

## Consequences

- One canonical selected-model vocabulary applies to every selection rule.
- Search objects no longer store two names for the same refitted objects.
- Global optimum diagnostics remain available through the non-estimator `best_*` attributes.
- Numerical evaluation, candidate selection, refitting, delegation, and serialization behavior are
  unchanged apart from the removed public aliases.
