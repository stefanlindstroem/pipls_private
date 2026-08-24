# Decision 0162: complete Pulp biplot helper

## Status

Accepted. Patch 0162A is implemented; Patch 0162B remains pending.

## Context

Decision 0161 established optional `textalloc` layout for the maintained Pulp score-loading biplot,
with other predictor labels and exact predictor-arrow shafts as the only allocator obstacles. The
initial implementation exposed that policy through a label-only helper. Review showed that this is
not the useful abstraction for either readers or maintainers: users want to create a biplot, while
label allocation is only one rendering detail.

The maintained example and documentation renderer should therefore share complete local biplot
helpers. The tutorial can then present the numerical `biplot_coordinates()` call followed by one
local plotting call, while hiding optional-dependency handling and annotation-layout details.

## Decision

Replace the label-only helper with one example-local Pulp biplot module containing:

- a simple Matplotlib renderer that draws sample scores, predictor arrows, axes, and predictor labels
  fixed at their endpoints;
- a `textalloc` renderer that draws the identical biplot geometry and allocates predictor labels
  against other predictor labels and the exact predictor-arrow shafts;
- a dispatcher that selects the `textalloc` renderer when the optional dependency is available and
  otherwise selects the simple renderer.

Both renderers must consume the immutable `BiplotCoordinates` result. Neither may alter the
numerical coordinates. Sample-score points remain deliberately excluded from the `textalloc`
obstacle set.

Increase the maintained predictor-label font size from 8 to 9 points. Tighten label placement around
the predictor endpoints by setting the `textalloc` placement-distance interval to 75% of the
previous default envelope: `min_distance=0.01125` and `max_distance=0.15`. Keep the Matplotlib axis
margin at `0.1`; that margin controls plot extent rather than label-to-endpoint placement.

Patch 0162A introduces the complete helper, migrates Example 04 and the Pulp tutorial renderer to
it, and adds focused tests for the simple path, the line-aware `textalloc` path, the dispatcher, the
font size, and the tighter placement envelope.

Patch 0162B will simplify the tutorial presentation so only the biplot-coordinate computation and
local helper call are shown, update the explanatory prose, regenerate and visually qualify the
maintained biplot, and close this decision.

## Consequences

- `pipls.inspection.biplot_coordinates()` remains the reusable package API; plotting stays
  caller-owned under Decision 0083.
- Example 04 and the documentation renderer share one complete biplot implementation rather than a
  label-only helper.
- The optional-dependency fallback is now expressed at the complete-biplot level.
- The `textalloc` collision contract from Decision 0161 is unchanged: other predictor labels and
  predictor-arrow shafts are avoided; sample scores are not obstacles.
- The maintained predictor labels are one point larger and remain closer to their predictor
  endpoints than under the previous placement envelope.
