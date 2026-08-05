# Decision 0061: example-owned report composition

## Status

Accepted and implemented.

## Context

The real-data report composer already owned Pi-PLS factor and prediction-diagnostic panels, but
scores, biplots, loadings, coefficients, and observation diagnostics initially relied partly on
package plotters. The report therefore did not apply caller-owned composition consistently. The
final data-first rendering policy in Decision 0083 removes the package plotting surface entirely.

## Decision

The example layer creates every report figure and axis directly from immutable numerical results,
adds legends and figure-level titles, writes PDF pages, and closes figures.

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

Repository tests enforce caller-owned rendering, direct use of numerical result fields, and no
runtime plotting module. Examples own legends, panels, layout, display, saving, and closing.
Example tests enforce that report composition and all figure construction remain in the caller.

## Consequences

The numerical inspection surface is uniform and reusable. Package code owns validated arrays and
scientific transformations; examples and user applications own every chart and presentation
choice.
