# Decision 0113: derive weighted response directions

## Status

Accepted and implemented.

## Context

`PiPLSDisplayFactors` stored `response_directions`, `dilation`, and an independently supplied
`weighted_response_directions` array. The last quantity is completely determined by the first two:

\begin{equation}
QD = Q\,\mathrm{diag}(d),
\end{equation}

or, in the public column-oriented representation, by elementwise multiplication of each response
direction column by its dilation value. The constructor therefore accepted redundant state and
then recomputed the same quantity solely to verify consistency.

The public `weighted_response_directions` attribute is useful for inspection and plotting and must
remain available. Direct construction and helper-produced results must also continue rejecting a
product that cannot be represented as finite float64.

## Decision

`PiPLSDisplayFactors` stores only the independent arrays:

- `predictor_directions`;
- `dilation`;
- `response_directions`.

`weighted_response_directions` remains a public read-only property. Each access derives $QD$ from
the validated `response_directions` and `dilation` arrays using the existing checked finite-product
operation. It is no longer a constructor argument or independently serialized field.

Direct construction still checks that the derived product is representable, so an invalid result
cannot be created and fail only later during plotting. `pipls_display_factors()` delegates that
check to the result constructor rather than forming and passing a redundant array.

This decision supersedes the independent-storage requirements in Decisions 0088 and 0094. It does
not change factor signs, values, attribute names, array shapes, or the regression-map identity.

## Consequences

- `factors.weighted_response_directions` remains available to existing inspection and rendering
  code as a defensive read-only array.
- The public constructor has three independent array arguments rather than four mutually constrained
  arguments.
- Inconsistent stored $Q$, $d$, and $QD$ states are impossible by construction.
- Pickle reconstruction serializes the three independent arrays and derives $QD$ after restoration.
- The package remains pre-release, so no compatibility constructor is retained.
