# Decision 0080: direct prediction-diagnostic rendering

## Status

Accepted and implemented.

## Context

`PredictionDiagnostics` already exposes the standardized observed, predicted, residual, and
response-wise RMSE arrays needed for the maintained diagnostic figures. The former convenience
plotters shortened examples but hid which arrays were displayed and which graphical choices were
made.

## Decision

Tutorials and numbered examples render prediction diagnostics directly with Matplotlib from:

- `observed_standardized`;
- `predicted_standardized`;
- `residual_standardized`;
- `standardized_rmse`;
- `prediction_kind`.

The public functions `plot_observed_vs_predicted()`, `plot_residuals_vs_predicted()`, and
`plot_standardized_rmse()` are removed immediately. Identity and zero-residual reference lines,
response subsets, labels, limits, legends, pagination, saving, and closing are caller-owned.

`PredictionDiagnostics` remains the stable numerical interface. The remaining plotting functions
stay temporarily until later plotting-migration patches.

## Consequences

- Programming users can see and modify every diagnostic plotting decision.
- Tutorial and example figures continue to use standardized arrays with explicit provenance.
- Tobacco retains deterministic response pagination and existing PDF filenames.
- The package plotting surface becomes smaller without changing estimator or diagnostic numerics.
