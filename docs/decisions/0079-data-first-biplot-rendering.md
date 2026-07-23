# Decision 0079: data-first biplot rendering

## Status

Accepted and implemented.

## Context

Component paths are taught as immutable numerical results rendered with ordinary Matplotlib, while
fitted-model figures were still taught through package-owned plotters. The score-loading biplot is
the only maintained chart that requires a nontrivial numerical preparation step because score and
loading coordinates need a declared balancing convention.

## Decision

Keep `biplot_coordinates()` and `BiplotCoordinates` as the package-owned numerical interface.
Remove `plot_biplot()` from `pipls.plotting`. The Pulp example and tutorial renderer draw samples,
loading arrows, labels, reference lines, and axes directly with Matplotlib.

Use `adjustText` as an optional examples, documentation, and development dependency. Maintained
biplots call `adjust_text()` only after the axis has been fully configured, use a fixed iteration
limit, and disable the experimental crossing-prevention option. Pi-PLS does not wrap `adjustText`.

## Consequences

Programming users see every biplot field and graphical decision. The balancing calculation remains
stable and reusable outside Matplotlib. `adjustText` is not a runtime dependency, and automatic
label placement remains an optional heuristic rather than a package guarantee.

This is the first increment in the plotting migration. Prediction-diagnostic plots are next.
