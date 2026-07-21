# Decision 0061: example-owned report composition

## Status

Accepted and implemented.

## Context

Decisions 0058--0060 made every public package plotter a one-axis chart primitive. The real-data
report composer already owned the Pi-PLS factor and prediction-diagnostic panels, but scores,
biplots, loadings, coefficients, and observation diagnostics still relied partly on standalone
figures created by package plotters. The report therefore did not apply the ownership boundary
consistently.

## Decision

The example layer creates every report figure and axis, passes an explicit `ax` to every package
plotter, adds legends and figure-level titles, writes PDF pages, and closes figures.

The maintained reports use these page groups:

- a $2\times2$ Pi-PLS factor page for $P$, $D$, $Q$, and $QD$;
- a $1\times3$ prediction-diagnostic page for each response group;
- one latent-model page chosen by the example content:
  - Pulp: scores, score-loading biplot, X loadings, and Y loadings in a $2\times2$ panel;
  - Sugarcane: scores, spectral X loadings, and Y loadings in a $1\times3$ panel;
  - Tobacco: scores, spectral X loadings, Y loadings, and observation diagnostics in a
    $2\times2$ panel;
- one full-width coefficient page for each response group.

Coefficient curves remain full-width because the physical predictor axis benefits from horizontal
space. Canonical CSV tables and numerical results are unchanged.

Repository tests enforce that public package plotters accept `ax`, create at most one standalone
axis through the shared resolver, and perform no legend, panel, layout, display, saving, or closing
operations. Example tests enforce that the report composer owns all figure construction and passes
an axis to every package plotter.

## Consequences

The plotting surface is uniform and reusable. Package code owns chart contents; examples and user
applications own presentation composition. The pre-release plotting-refinement phase is complete,
and release preparation may resume.
