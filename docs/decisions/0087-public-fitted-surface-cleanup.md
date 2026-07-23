# Decision 0087: public fitted-surface cleanup

## Status

Accepted and implemented.

## Context

The fixed estimator and path selector retained several fitted attributes that duplicated more
meaningful public results or exposed implementation bookkeeping. `PiPLSRegression` published an
internal response scale used only by the default scorer and exact `x_weights_`/`y_weights_` aliases
of its rotation arrays. `PiPLSPathCV` published validated input grids, adaptive-search batch
history, candidate counters, direct-rank parameter aliases, and flat OOF attributes already stored
in `component_path_`, `cv_results_`, `best_*`, or `validation_report_`.

These names enlarged completion lists and generated reference pages without adding independent
numerical information.

## Decision

`PiPLSRegression` retains scores, loadings, rotations, coefficients, intercepts, preprocessing
statistics, rank information, and `decomposition_`. The response scale required by the package
scorer is private fitted state. Exact `x_weights_` and `y_weights_` aliases are removed; users use
`x_rotations_` and `y_rotations_`.

`PiPLSPathCV` retains:

- standard `cv_results_`, `best_index_`, `best_score_`, `best_params_`, and refit attributes;
- direct `best_n_components_` and `best_predictor_rank_` values;
- `component_path_` and `predictor_rank_profile()`;
- `validation_report_`;
- `path_search_exhaustive_`;
- fitted data, split, scorer, and rank-limit diagnostics.

Optional OOF predictions, coverage counts, pooled OOF $R^2$, and the selected direct rank pair live
only in `validation_report_`. Validated input grids, candidate counts, adaptive batch history,
search-method echoes, `best_pipls_params_`, and the flat OOF aliases are not public fitted state.
`cv_results_` remains the complete candidate-level record.

## Consequences

- The public fitted surface contains independent results rather than exact aliases or execution
  traces.
- Pipeline-prefixed `best_params_` remains scikit-learn compatible, while direct selected ranks are
  available through `best_n_components_`, `best_predictor_rank_`, and `validation_report_`.
- Users inspect evaluated ranks through `cv_results_` or `predictor_rank_profile()` and component
  policies through `component_path_`.
- Adaptive-search implementation details remain private and may change without expanding the
  compatibility contract.
- Existing pre-release code using removed fitted aliases must migrate to the retained result
  objects and attributes.
