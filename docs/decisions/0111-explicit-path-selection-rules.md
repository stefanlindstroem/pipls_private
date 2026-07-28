# Decision 0111: explicit path selection rules and selected-model refitting

## Status

Accepted.

## Context

`PiPLSRegression` correctly fits one explicit `(n_components, predictor_rank)` pair, while
`PiPLSPathCV` owns cross-validated selection. Before this decision, `PiPLSPathCV(refit=True)` could
only refit the globally best evaluated pair under the configured scorer. The component path could
also return a stored one-standard-error recommendation, but applying that declared rule required a
separate estimator construction and fit.

Some applications use a fully predefined protocol, for example adaptive predictor-rank selection
at every component count followed by the 1-SE component rule. Such a protocol can be executed in
one call without making selection hidden, provided the rule is explicit and all evaluated and
selected results remain inspectable.

A second wrapper would duplicate the path selector's candidate evaluation, split handling,
preprocessing, result records, OOF generation, and refit delegation. The missing responsibility is
therefore final-row selection inside `PiPLSPathCV`, not another model-building class.

## Decision

`PiPLSPathCV` adds the constructor parameter

```python
selection_rule="best_score"
```

with the accepted values:

- `"best_score"`: choose the globally best evaluated pair under the configured scorer;
- `"one_standard_error"`: choose the row returned by
  `component_path_.one_standard_error_result()`.

The default remains `"best_score"`, and `refit=False` remains the default. The fixed estimator
continues to perform no selection.

Standard `best_index_`, `best_score_`, `best_params_`, `best_n_components_`, and
`best_predictor_rank_` always describe the global configured-score optimum, regardless of the final
selection rule. The final declared choice is represented separately by:

- `selected_result_`, the immutable stored `PiPLSComponentResult` row;
- `selected_params_`, the estimator parameter mapping for that row.

`validation_report_` and optional ordered OOF predictions represent `selected_result_`. With
`refit=True`, the selector fits that pair on all supplied data and exposes `selected_estimator_` and
`selected_pipls_`; delegated prediction, transformation, reconstruction, scoring, and feature-name
methods use the selected estimator.

For the default `selection_rule="best_score"`, `best_estimator_` and `best_pipls_` remain compatibility
aliases of the selected refit. They are not exposed for `selection_rule="one_standard_error"`, where
calling the refitted model “best” would be misleading. The selector refits one final model, not both
the global optimum and the declared recommendation.

The 1-SE rule retains the contract of Decision 0107. It operates on the stored component path,
uses the minimum-row fold-based standard error, and retains the predictor rank already selected
conditionally for the chosen component count. With a nondefault scorer, those conditional ranks
follow that scorer even though the component rule uses the stored response-standardized CV-MSE.
At least two validation splits are required.

## Consequences

A predeclared automated workflow is possible without adding hidden selection to
`PiPLSRegression`:

```python
search = PiPLSPathCV(
    search_method="auto",
    selection_rule="one_standard_error",
    refit=True,
).fit(X, Y)

model = search.selected_pipls_
```

The complete selection record remains available through `cv_results_`, `component_path_`,
`predictor_rank_profile()`, the global `best_*` attributes, `selected_result_`, and
`validation_report_`. Post-hoc scientific judgment after inspecting a path remains an explicit
two-stage workflow. Decisions 0102 and 0107 retain their default and result-method contracts but are
extended by this explicit selected-model orchestration.
