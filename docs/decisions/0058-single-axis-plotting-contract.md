# Decision 0058: single-axis plotting contract

## Status

Accepted and implemented for the existing single-chart plotting functions. The two composite
plotters remain temporary migration targets.

## Context

The plotting surface mixed atomic PLS-style functions with package-owned multi-panel figures.
Atomic functions also always created their own figures and returned one-entry axis dictionaries,
which prevented callers from composing them naturally with ordinary Matplotlib layouts.

A reusable plotting layer should provide scientific chart primitives. Applications and examples
should decide whether those charts appear alone, in a panel, or in a multipage report.

## Decision

- A public single-chart plotting function draws exactly one chart on exactly one Matplotlib `Axes`.
- It accepts `ax=None` and returns `(figure, axis)`.
- With `ax=None`, it creates one figure containing one axis.
- With a supplied axis, it does not clear the axis, change the surrounding figure, or create another
  figure.
- `figsize` applies only to standalone creation and is invalid together with `ax`.
- Plotters may provide concise semantic axis labels and an axis title. Callers may replace or remove
  them through the returned axis.
- Multi-series artists carry labels, but plotters do not create legends. Callers own legend
  creation, placement, and styling.
- Callers own subplot grids, mosaics, figure-level titles, layout adjustment, saving, display, and
  closing.
- `plot_pipls_decomposition()` and `plot_prediction_diagnostics()` remain temporary composite
  exceptions. They must be replaced by single-chart functions before the first release.

This contract is implemented first for `plot_scores()`, `plot_x_loadings()`, `plot_y_loadings()`,
`plot_coefficients()`, `plot_biplot()`, and `plot_observation_diagnostics()`.

## Consequences

The same plotting function can be used as a standalone convenience or embedded in a caller-owned
panel. One-entry axis dictionaries disappear from the atomic API. The example layer adds legends
explicitly and continues to own PDF writing and closing. The next plotting increment splits the
Pi-PLS decomposition display into one function per factor quantity.
