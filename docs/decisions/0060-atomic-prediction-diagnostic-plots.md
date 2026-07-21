# Decision 0060: atomic prediction-diagnostic plots

## Status

Accepted and implemented.

## Context

`plot_prediction_diagnostics()` created a package-owned three-axis figure for observed versus
predicted responses, residuals versus predicted responses, and response-wise standardized RMSE.
This conflicted with the single-axis plotting contract in Decision 0058 and prevented callers from
choosing their own panel geometry and provenance placement.

The immutable `PredictionDiagnostics` object already provides all values and the explicit
`prediction_kind` needed by separate charts.

## Decision

Replace the composite function with three public plotting functions:

- `plot_observed_vs_predicted()` for standardized observations against standardized predictions;
- `plot_residuals_vs_predicted()` for standardized residuals against standardized predictions;
- `plot_standardized_rmse()` for response-wise standardized RMSE.

Each function follows Decision 0058: it draws one chart on one axis, accepts `ax=None`, returns
`(figure, axis)`, labels response series without creating a legend, and performs no panel, layout,
or file-output operations. Standalone calls include `PredictionDiagnostics.prediction_kind` in the
axis title by default. Callers composing panels may set `include_prediction_kind=False` and report
provenance once at figure level.

Remove `plot_prediction_diagnostics()` before the first release rather than retain a compatibility
wrapper. The package has no tagged public release, and retaining the composite would preserve the
design being corrected.

The real-data report helper creates its own $1\times3$ prediction panel, adds legends when several
responses are shown, places prediction provenance in the figure title, writes the PDF page, and
closes the figure.

## Consequences

All public package plotters now represent one chart and compose naturally with ordinary Matplotlib
layouts. `src/pipls/plotting.py` owns chart primitives; examples and applications own panel
composition and reporting. A final audit remains to apply this boundary consistently across all
maintained example pages and documentation.
