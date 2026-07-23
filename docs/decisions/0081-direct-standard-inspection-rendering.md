# Decision 0081: direct standard inspection rendering

## Status

Accepted and implemented.

## Context

`LatentStructure` and `ObservationDiagnostics` already expose the numerical arrays required for
scores, X loadings, Y loadings, regression coefficients, and raw observation diagnostics. The
former convenience plotters shortened examples but hid variable selection, physical coordinates,
chart type, grouping, labels, and reference lines.

## Decision

Maintained examples render standard PLS-family inspection quantities directly with Matplotlib from:

- `LatentStructure.x_scores`;
- `LatentStructure.x_loadings`;
- `LatentStructure.y_loadings`;
- `LatentStructure.coefficients`;
- `ObservationDiagnostics.score_distance`;
- `ObservationDiagnostics.x_reconstruction_residual`.

The public functions `plot_scores()`, `plot_x_loadings()`, `plot_y_loadings()`,
`plot_coefficients()`, and `plot_observation_diagnostics()` are removed immediately. Dataset-specific
rendering choices remain visible in each example: grouped bars for named variables, line plots for
physical spectral coordinates, and direct scatter plots for scores and observation diagnostics.

The immutable inspection results remain the stable numerical interface. Pi-PLS factor plotters stay
temporarily until the next plotting-migration patch.

## Consequences

- Programming users can see which arrays and coordinates define each standard inspection figure.
- Spectral coordinate order, response pagination, labels, legends, saving, and closing remain
  example-owned.
- Existing final PDF filenames and Tobacco page structure remain unchanged.
- The package plotting surface is reduced to the four Pi-PLS factor renderers.
