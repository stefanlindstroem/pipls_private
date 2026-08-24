# Decision 0163: response-wise OOF R² diagnostics

## Status

Accepted; patches 0163A--B establish the presentation contract and migrate Pulp. Patch 0163C
remains active.

## Context

The maintained Pulp, Sugarcane, and Tobacco workflows currently display response-wise standardized
RMSE as the scalar summary in their selection-conditioned out-of-fold (OOF) diagnostics. The
underlying `PredictionDiagnostics` result already exposes both `standardized_rmse` and
`response_r2`, so changing the visible diagnostic does not require a numerical or public-API
extension.

Response-wise OOF R² is more immediately interpretable for many readers: 1 denotes exact agreement,
0 is the observed-mean reference, and negative values indicate predictions worse than that
reference. Standardized RMSE remains scientifically useful and remains the response-scale-normalized
diagnostic most directly connected to the response-standardized loss used during path selection.
The change is therefore a presentation choice, not removal or deprecation of standardized RMSE.

OOF R² in these workflows is computed from the row-ordered OOF predictions supplied to
`prediction_diagnostics()`. Under repeated cross-validation, repeated held-out predictions are
combined per observation by `oof_report()` before the response-wise R² is calculated. It is not an
average of fold-wise R² values.

## Decision

Make response-wise selection-conditioned OOF R² the preferred visible scalar response diagnostic in
Examples 04--06 and Tutorial 3.

Keep all of the following unchanged:

- `PredictionDiagnostics.standardized_rmse` and `PredictionDiagnostics.response_r2` remain public
  numerical diagnostics;
- response-standardized MSE remains the model-selection loss;
- Quick Start / Example 01 retains its fitted-value standardized-RMSE summary because it is not the
  selection-conditioned OOF diagnostic covered by this decision;
- pooled OOF R² remains distinct from the response-wise values shown in the bar plots.

Every maintained response-wise R² bar plot must use the following y-axis contract:

- the upper limit is exactly `1.0`;
- the lower limit is never greater than `0.0`;
- if every displayed R² value is nonnegative, the lower limit is exactly `0.0`;
- if any displayed R² value is negative, the lower limit extends below the most negative value so
  that negative predictive performance is not clipped;
- a horizontal reference line at R² = 0 should be drawn when the plot is migrated.

The example-local `response_r2_ylim()` helper owns the numerical y-limit rule. It uses a small
5%-of-range downward padding below a negative minimum and otherwise returns `(0.0, 1.0)`. Plotting
remains caller-owned under Decision 0083.

Patch 0163A establishes this contract, adds the shared example-local y-limit helper, replaces the
old source-level test that treated standardized RMSE and R² as sharing one unit-interval contract,
and adds focused helper tests. It does not change visible example output.

Patch 0163B migrates Tutorial 3 and Example 04 from the visible OOF standardized-RMSE bar plot to
response-wise OOF R², records the response-wise OOF values in the generated tutorial manifest,
applies the shared R² y-limit helper to both OOF and final-fit Pulp R² plots, and keeps their
prediction provenance explicit. The generated OOF scalar asset is renamed to
`oof_response_r2.svg`; standardized RMSE remains available numerically through
`PredictionDiagnostics`.

Patch 0163C migrates the analogous Sugarcane and Tobacco OOF panels, synchronizes documentation and
release notes, qualifies all maintained examples/documentation, and closes the decision.

## Consequences

- Readers see response-wise OOF R² in the maintained real-data predictive-diagnostic figures.
- Standardized RMSE remains available for numerical inspection and explanation of the relationship
  to response-standardized model-selection loss.
- OOF R² and final-fit R² must be labelled by provenance so fitted-value agreement is not confused
  with held-out prediction evidence.
- Negative R² values are first-class diagnostic results and may never be hidden by a hard zero
  lower axis limit.
- The public package API and numerical model-selection behavior are unchanged.
