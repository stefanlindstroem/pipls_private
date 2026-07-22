# Decision 0066: immutable component-path API

## Status

Accepted and implemented.

## Context

`PiPLSPathCV` already exposes the complete scikit-learn-style candidate table through
`cv_results_`. Its former concise component view duplicated selected values across a dictionary of
aligned arrays and two additional component-keyed dictionaries. Accessing one component choice
therefore required table construction, indexing syntax, or synchronization among several fitted
attributes.

The package otherwise uses frozen result objects with direct attribute access for decomposition,
validation, datasets, and model inspection. The concise path should follow the same convention
without changing the standard `cv_results_` surface.

## Decision

Add two top-level public frozen result types:

- `PiPLSComponentPath`, containing aligned read-only arrays named `n_components`,
  `predictor_rank`, `predictor_rank_policy`, `mean_test_score`, `cv_mse_mean`,
  `cv_mse_fold_sd`, and `n_splits`;
- `PiPLSComponentResult`, containing the corresponding Python scalar values for one component
  count.

A fitted `PiPLSPathCV` exposes the concise result as `component_path_`.
`component_path_.for_n_components(h)` returns the unique scalar result for an evaluated component
count and raises a `ValueError` that lists the available counts when `h` was not evaluated.

All arrays are defensive copies, one-dimensional, equal in length, and read-only. Component counts
are positive, unique, and strictly ascending. Predictor ranks remain admissible for their aligned
component counts, policies remain explicit, fold counts are positive, and fold standard deviations
are nonnegative.

The former concise mapping and separate component-keyed best-rank and best-score fitted attributes
are removed before release. `cv_results_`, `best_index_`, `best_score_`,
`best_n_components_`, `best_predictor_rank_`, `best_params_`, and `validation_report_` remain.

## Consequences

- Ordinary numerical workflows inspect path arrays directly and retrieve a selected pair without
  pandas or mapping syntax.
- Custom scoring remains visible through `mean_test_score`; response-standardized CV-MSE remains a
  separate diagnostic through `cv_mse_mean` and `cv_mse_fold_sd`.
- `cv_results_` retains the established candidate-level dictionary-of-arrays convention and remains
  the detailed path surface.
- Existing numbered examples and documentation consumers migrate to `component_path_` in this
  increment while retaining their current generated artifacts. Removal of CSV intermediates is a
  subsequent example-layer increment.
- Pickling preserves the frozen result objects and read-only arrays.
