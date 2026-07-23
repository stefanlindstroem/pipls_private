# Decision 0012: scikit-learn and PLS-style public API alignment

## Status

Accepted and implemented, with selection-result ownership refined by Decision 0039: search
attributes belong to `PiPLSPathCV`, while `PiPLSRegression` is a direct estimator.

## Context

Pi-PLS has a fixed-model estimator and a pipeline-aware path meta-estimator. Programming users
should be able to apply ordinary scikit-learn expectations and transfer familiar
`PLSRegression` workflows without learning a second set of method, feature-name, coefficient, or
selection-result conventions. Pi-PLS also exposes factorization matrices that have no direct
`PLSRegression` counterpart.

## Decision

`PiPLSRegression` is a multi-output regressor and transformer. It uses estimator-aware data
validation, records `n_features_in_` and `feature_names_in_`, supports `get_feature_names_out` and
`set_output`, and mirrors the PLS-style `predict(X, copy=True)`, `transform(X, y=None, copy=True)`,
and tuple-valued `fit_transform(X, y)` behavior.

The standard fitted surface includes `x_weights_`, `y_weights_`, `x_loadings_`, `y_loadings_`,
`x_scores_`, `y_scores_`, `x_rotations_`, `y_rotations_`, `coef_`, and `intercept_`. In Pi-PLS,
the direct orthogonal score maps are both weights and rotations. Loadings are separately computed
least-squares reconstruction coefficients. `n_iter_` is omitted because Pi-PLS uses a closed-form
SVD construction rather than an iterative NIPALS loop.

Pi-PLS-specific factorization and numerical diagnostics are canonicalized in the public frozen
`PiPLSDecomposition` object exposed as `decomposition_`. Decision 0040 removes duplicate symbolic
and diagnostic top-level aliases so the structured result is the single method-specific source.

`PiPLSRegression` is a direct estimator and exposes no search-result attributes.
`PiPLSPathCV` owns `cv_results_`, `best_params_`, `best_index_`, `best_score_`, and the conditional
component-path results. `rank_test_score` uses minimum ranks for tied scores, while the explicit
Pi-PLS complexity rule selects the smaller admissible model among score ties. Every tolerant
comparison is made against one reference score: rank 1 is the selector's direct tie set around the
maximum, and each lower rank group is anchored to its leading score rather than chained through
adjacent near-ties.

`PiPLSPathCV` is a regressor, transformer, and meta-estimator. It preserves indexable input
containers inside folds so pandas column names and column-selecting pipelines continue to work.
It exposes the complete selected estimator as `best_estimator_` and the fitted nested
`PiPLSRegression` as `best_pipls_`, with direct Pi-PLS parameter values in
`best_pipls_params_`. It does not flatten nested coefficients onto the path object because those
coefficients may be defined after learned preprocessing.

Both public classes return uniformly averaged R2 from `score`. `best_score_` remains the
cross-validation selection score and may therefore use a different metric.

## Consequences

- Direct estimator and path workflows share prediction, transformation, scoring, feature-name,
  decomposition, and selected-model access patterns.
- `copy=False` now has the same observable fit-time in-place preprocessing semantics expected by
  PLS users for writable floating NumPy arrays.
- Generic scikit-learn transformer checks do not recognize third-party cross-decomposition class
  names when `fit_transform(X, y)` returns both X and y scores. Tests declare only those
  tuple-contract checks as expected failures; all remaining common estimator checks pass.
- Composite path users access fitted Pi-PLS internals through `best_pipls_`, avoiding misleading
  top-level coefficient aliases.
