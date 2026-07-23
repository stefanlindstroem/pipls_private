# Decision 0082: direct Pi-PLS factor rendering

## Status

Accepted and implemented.

## Context

`PiPLSDisplayFactors` already exposes defensive read-only arrays for the display-signed predictor
and response directions, dilation values, and weighted response directions. The remaining
`pipls.plotting` functions shortened factor figures but hid categorical positions, grouping widths,
component selections, physical predictor coordinates, labels, and chart choices. After the biplot,
prediction-diagnostic, and standard PLS-family migrations, these four functions were the only
remaining plotting surface.

## Decision

Maintained examples render $P$, $D$, $Q$, and $QD$ directly from:

- `PiPLSDisplayFactors.predictor_directions`;
- `PiPLSDisplayFactors.dilation`;
- `PiPLSDisplayFactors.response_directions`;
- `PiPLSDisplayFactors.weighted_response_directions`.

The examples use ordinary Matplotlib bars or lines and own component subsets, grouping offsets,
physical spectral coordinates, labels, legends, titles, saving, and closing. The public
`pipls.plotting` module, its four factor renderers, `PredictorStyle`, its generated API page, and the
`plot` optional dependency extra are removed immediately. No deprecated aliases or compatibility
wrappers are retained before the first release.

Matplotlib and `adjustText` remain optional dependencies under the `examples`, `docs`, and `dev`
extras. The runtime package imports neither dependency.

## Consequences

- `pipls.inspection` is the sole fitted-model analysis submodule.
- Programming users see the numerical factor arrays and every rendering choice directly.
- Existing numbered-example PDF filenames and panel organization remain unchanged.
- Source distributions and wheels contain no plotting module.
- Final plotting-policy cleanup and structural enforcement remain for the next patch.
