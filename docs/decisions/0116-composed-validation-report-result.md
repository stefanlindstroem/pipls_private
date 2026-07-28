# Decision 0116: compose validation reports from the selected component result

## Status

Accepted and implemented.

## Context

`PiPLSValidationReport` independently stored the selected component count, predictor rank, split
count, mean configured score, and mean response-standardized CV-MSE. The same values were already
stored in `PiPLSSearchCV.selected_result_` as one validated immutable `PiPLSComponentResult`.
`PiPLSSearchCV.fit()` therefore copied five selected-row scalars into the report, and direct report
construction could represent a validation summary that disagreed with the selected component result
used elsewhere by the search.

The report-specific state is the validation provenance, leave-one-out classification, optional OOF
predictions and coverage counts, and optional pooled OOF $R^2$. The selected path row is an existing
public value object and should be composed rather than flattened into duplicate state.

## Decision

`PiPLSValidationReport` accepts and stores one validated `selected_result` of type
`PiPLSComponentResult`. It no longer accepts or serializes independent `n_components`,
`predictor_rank`, `n_splits`, `mean_test_score`, or `mean_response_standardized_mse` constructor
arguments.

The existing public read names remain as properties:

- `n_components` forwards to `selected_result.n_components`;
- `predictor_rank` forwards to `selected_result.predictor_rank`;
- `n_splits` forwards to `selected_result.n_splits`;
- `mean_test_score` forwards to `selected_result.mean_test_score`;
- `mean_response_standardized_mse` forwards to `selected_result.cv_mse_mean`.

`PiPLSSearchCV.validation_report_` stores the same immutable object exposed as
`PiPLSSearchCV.selected_result_`. OOF array, coverage, provenance, leave-one-out, pooled-$R^2$, and
pickle validation remain owned by `PiPLSValidationReport`.

Direct construction requires a `PiPLSComponentResult` and rejects another object with `TypeError`.
This decision supersedes the independently stored selected-row scalar fields in Decisions 0005,
0093, and 0111. It does not change selection, scoring, OOF generation, report convenience names, or
numerical results.

## Consequences

- One immutable object is authoritative for the selected component-path row.
- Validation reports cannot disagree with their selected result on ranks, scores, or split count.
- Existing code that reads `report.n_components`, `report.predictor_rank`, `report.n_splits`,
  `report.mean_test_score`, or `report.mean_response_standardized_mse` continues to work.
- Direct report construction becomes smaller and uses `selected_result=...`.
- Pickle reconstruction contains the selected result once together with report-specific state.
