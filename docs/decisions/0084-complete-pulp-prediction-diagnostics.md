# Decision 0084: complete Pulp prediction diagnostics

## Status

Accepted and implemented.

## Context

The focused Pulp tutorial displayed observed-versus-predicted and standardized-RMSE figures while
its checked example snippet created three prediction-diagnostic axes. The undisplayed middle axis
contained standardized residuals versus standardized predictions. This made the tutorial code and
figure inventory inconsistent and left `residual_standardized` unexplained by a visible chart.

## Decision

Retain the complete standard prediction-diagnostic trio in Tutorial 2:

```text
observed_vs_predicted.svg
residuals_vs_predicted.svg
standardized_rmse.svg
```

Generate the residual figure directly from `PredictionDiagnostics.predicted_standardized` and
`PredictionDiagnostics.residual_standardized`, with a zero reference line and the same three
detailed Pulp responses used by the observed-versus-predicted figure. Preserve the complete
three-panel numbered-example report unchanged.

## Consequences

- The displayed tutorial figures now correspond to all three axes in the checked prediction snippet.
- Tutorial 2 contains seven representative SVG figures rather than six.
- The residual plot remains descriptive and selection-conditioned; it is not a formal variance or
  calibration analysis.
- Generated tutorial assets remain ignored working-tree products and are validated by renderer and
  manifest tests rather than by the generic static-link check.
- The data-first rendering policy remains unchanged.
