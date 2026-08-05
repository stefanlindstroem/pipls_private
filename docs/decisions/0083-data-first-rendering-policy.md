# Decision 0083: final data-first rendering policy

## Status

Accepted and implemented.

## Context

A sequence of rendering migrations moved biplots, prediction diagnostics, standard PLS-family
inspection, and Pi-PLS factors from package-owned plotters to direct Matplotlib construction. After
the final plotting module was removed, the repository still needed one durable policy covering
documentation, optional dependencies, examples, distribution checks, and structural enforcement.

## Decision

Pi-PLS owns numerical modeling and immutable inspection results. It does not own a plotting
submodule or public `plot_*` convenience functions. Maintained tutorials and numbered examples
render named result arrays with ordinary Matplotlib so scientific coordinates, selections, labels,
legends, layout, and output remain visible and caller-controlled.

`biplot_coordinates()` remains package-owned because balancing score and loading coordinates is a
numerical operation. Annotated biplots use Matplotlib arrows and text artists. The optional external
`adjustText` package may reposition those labels after final axis configuration. Automatic label
placement is heuristic and is not part of the Pi-PLS numerical contract.

Matplotlib and `adjustText` remain optional dependencies under the `examples`, `docs`, and `dev`
extras. The runtime package imports neither package. No example support helper may hide chart
construction or read serialized analytical results for plotting.

## Consequences

- `pipls.inspection` is the public numerical analysis surface.
- Users may render results with Matplotlib, another graphics library, or no graphics system.
- Tutorials expose the result fields used for each maintained chart.
- Structural tests protect module absence, optional dependency boundaries, direct rendering,
  biplot-coordinate use, and runtime imports without graphics dependencies.
- Exact visual styling, automatically adjusted label positions, and incidental artist counts are
  not compatibility contracts.
