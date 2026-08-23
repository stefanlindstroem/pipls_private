# Decision 0161: line-aware text allocation for annotated biplots

## Status

Accepted; implementation in progress.

## Context

Decision 0160 made `adjustText` optional and preserved a plain-Matplotlib fallback for the Pulp
biplot. The maintained figure nevertheless needs a stronger annotation-layout contract: predictor
labels should avoid other predictor labels and the predictor-arrow shafts themselves. Approximating
those shafts with sampled repulsion points does not express the geometry directly and did not give
reliable placement in review.

`textalloc` accepts plotted line segments as explicit obstacles through `x_lines` and `y_lines` and
already avoids previously allocated text boxes. That is a better match for the Pulp biplot, where
each predictor arrow is the segment from the origin to one predictor endpoint.

## Decision

Migrate the maintained Pulp annotation layout from `adjustText` to `textalloc` in three increments.

For the Pulp biplot, the allocator-owned collision set is deliberately narrow:

- other predictor labels are avoided by the allocator;
- each predictor-arrow shaft is supplied as an exact two-point line obstacle from `(0, 0)` to its
  predictor endpoint;
- sample-score points are **not** supplied as obstacles;
- labels are all retained;
- extra connector lines from predictor endpoints to moved labels are disabled initially because the
  predictor arrows already provide the visual association.

The layout remains example/documentation behavior rather than runtime package behavior. If
`textalloc` is unavailable, the maintained workflow must remain executable and leave labels at
their ordinary Matplotlib endpoint positions. Absence of the optional package itself may be handled
as a fallback; unrelated import failures from an installed package must continue to propagate.

Patch 0161A introduces `textalloc` alongside `adjustText` in the maintained extras, adds one
example-local allocation helper that represents predictor arrows as actual line segments, and
regression-tests the line-only obstacle contract. It does not change the rendered Pulp workflows.

Patch 0161B migrates Example 04 and the Pulp tutorial renderer to that helper and qualifies the
result visually and for deterministic documentation rendering.

Patch 0161C removes `adjustText` from the active dependency and documentation surface, records
Decision 0160 as superseded only with respect to the allocator choice while preserving its graceful
fallback boundary, and closes this decision.

## Consequences

- No plotting dependency enters the `pipls` runtime package.
- `textalloc` and `adjustText` temporarily coexist in the `examples`, `docs`, and `dev` extras while
  the replacement is qualified.
- The helper has no sample-coordinate parameter, preventing accidental repulsion from the score
  cloud in the maintained annotation policy.
- Predictor arrows are represented by their true line-segment geometry rather than sampled points.
- The plain-Matplotlib endpoint-label fallback remains the required behavior when the preferred
  allocator is unavailable.
