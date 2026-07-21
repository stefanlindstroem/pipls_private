# Decision 0059: atomic Pi-PLS factor plots

## Status

Accepted and implemented.

## Context

`plot_pipls_decomposition()` combined predictor directions, weighted response directions, and
component dilation in a package-owned three-row figure. It omitted the unweighted response
matrix $Q$, prevented callers from choosing their own panel geometry, and conflicted with the
single-axis plotting contract in Decision 0058.

The immutable `PiPLSDisplayFactors` object already provides the four quantities needed for
interpretation: $P$, $D$, $Q$, and $QD$.

## Decision

Replace the composite function with four public plotting functions:

- `plot_pipls_predictor_directions()` for columns of $P$;
- `plot_pipls_dilation()` for $d_k=D_{kk}$;
- `plot_pipls_response_directions()` for columns of $Q$;
- `plot_pipls_weighted_response_directions()` for columns of $QD$.

Each function follows Decision 0058: it draws one chart on one axis, accepts `ax=None`, returns
`(figure, axis)`, labels multi-series artists without creating a legend, and performs no layout or
file-output operations. Predictor directions retain explicit bar and physical-coordinate line
rendering. Response-side factors use grouped bars with caller-supplied response names.

Remove `plot_pipls_decomposition()` before the first release rather than preserve a compatibility
wrapper. The package has no tagged public release, and retaining the composite would preserve the
design being corrected.

The minimal and real-data examples create their own $2\times2$ factor panels, add legends and
figure-level titles, write PDFs, and close figures.

## Consequences

The public Pi-PLS plotting surface now represents all four factor quantities and is composable with
ordinary Matplotlib layouts. `src/pipls/plotting.py` owns chart primitives; examples own panel
composition. Decision 0060 applies the same boundary to prediction diagnostics.
