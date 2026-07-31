# Decision 0134: type-revealing public result properties

## Status

Accepted and implemented.

## Context

Several unreleased public result properties did not communicate their returned type or predicate
semantics clearly.

`PiPLSPredictorRankProfile.selected` returns a complete `PiPLSComponentResult`, but its name can be
read as a boolean or an unspecified selected value. `PiPLSValidationReport` exposes the same
response-standardized CV-MSE quantity as other path result records under the longer
`mean_response_standardized_mse` name. Its boolean properties `selection_conditioned` and
`complete_oof_coverage` are descriptive nouns rather than predicates.

The package remains unreleased at version `0.0.0`, so no published compatibility commitment
requires retaining these names.

## Decision

Rename the public properties as follows:

- `PiPLSPredictorRankProfile.selected` becomes `selected_result`;
- `PiPLSValidationReport.mean_response_standardized_mse` becomes `cv_mse_mean`;
- `PiPLSValidationReport.selection_conditioned` becomes `is_selection_conditioned`;
- `PiPLSValidationReport.complete_oof_coverage` becomes `has_complete_oof_coverage`.

The renamed properties return the same derived scalar result, score, and boolean values as before.
The candidate-table keys `mean_response_standardized_mse` and
`std_response_standardized_mse` in `PiPLSSearchCV.cv_results_` remain unchanged because they name
cross-validation table columns rather than result-record properties.

Use the new names throughout implementation, tests, examples, renderers, public documentation, and
active guide-layer contracts. Do not add aliases, fallback attribute lookup, or serialization
migration for the former unreleased names.

Historical decision records remain unchanged. This decision supersedes Decisions 0072 and 0115
only where they require the former predictor-rank-profile property spelling, and Decision 0116 only
where it requires the former validation-report convenience-property spellings. Their immutable
state, tie-breaking, composition, OOF, and numerical contracts otherwise remain in force.

## Consequences

- Predictor-rank-profile code states explicitly that the derived value is a result object.
- Validation-report CV-MSE naming matches `PiPLSComponentResult`, `PiPLSComponentPath`, and
  `PiPLSPredictorRankProfile`.
- Validation-report booleans read as predicates.
- Selection, scoring, CV-MSE values, OOF coverage, pickling, and immutable result state are
  unchanged.
- Existing pre-release code using the former property names must adopt the new names.
- No duplicate compatibility properties enlarge the public result surface.
