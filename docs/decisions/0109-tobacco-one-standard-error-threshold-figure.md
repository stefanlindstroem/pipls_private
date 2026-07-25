# Decision 0109: Tobacco one-standard-error threshold figure

## Status

Accepted.

## Context

Decision 0108 made example 07 the maintained application of
`PiPLSComponentPath.one_standard_error_result()`, but its component-path figure marked only the
recommended row. The horizontal threshold defining the rule was not visible, and the example
catalogue linked only indirectly to the general path-analysis material. That left the graphical
application incomplete, especially when the CV-MSE path has no clear elbow.

## Decision

`examples/07_tobacco_real_data.py` also obtains the stored minimum-CV-MSE row through
`minimum_cv_mse_result()` solely to construct the explanatory figure. It calculates the 1-SE
threshold from that row, marks the minimum row, draws the threshold as a horizontal dashed line, and
marks the row returned by `one_standard_error_result()`. The final fixed model remains determined by
`one_standard_error_result()`; the example does not implement a second selection calculation.

`docs/examples.md` contains a dedicated Tobacco one-standard-error subsection linking to the
path-analysis explanation and component-path API reference. `docs/path_analysis.md` links back to
that subsection and notes that the rule can be convenient when no clear elbow identifies a
component count.

This decision supersedes only Decision 0108's statement that the example does not call
`minimum_cv_mse_result()` and its label-only figure description. The recommendation methods, exact
stored-value comparisons, predictor-rank contract, and nonautomatic `PiPLSPathCV` behavior are
unchanged.

## Consequences

The maintained Tobacco figure now displays every quantity needed to understand the 1-SE
recommendation. The minimum-row lookup has a concrete presentational purpose and introduces no
additional model choice. Documentation readers can move directly between the rule and its complete
real-data application without promoting the method on the documentation home or tutorials.
