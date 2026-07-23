# Decision 0088: public result-record cleanup

## Status

Accepted and implemented.

## Context

Two immutable public records still exposed bookkeeping rather than independent user results.
`PiPLSDisplayFactors` returned the signs used internally to canonicalize factor columns even though
maintained workflows consume only the resulting display-signed factors. `PiPLSSyntheticTruth`
stored two zero loading blocks for effects that are structurally absent from the generator.

Generated API pages also displayed full dataclass constructor signatures for fitted and derived
result records. Those objects are normally obtained from estimators or numerical helper functions;
the prominent constructors obscured how users actually acquire and inspect them.

## Decision

`PiPLSDisplayFactors` retains only:

- `predictor_directions`;
- `dilation`;
- `response_directions`;
- `weighted_response_directions`.

The deterministic sign calculation remains internal to `pipls_display_factors()` and is verified
through the returned factor arrays rather than a public `component_signs` field.

`PiPLSSyntheticTruth` retains only latent scores, loading blocks that contribute to the generated
predictor or response signal, signal/noise matrices, observed-variable scales, and latent
strengths. The structurally impossible `x_response_specific_loadings` and
`y_predictor_specific_loadings` zero arrays are removed.

Generated reference pages suppress constructor signatures for immutable records returned by
estimators, searches, generators, or inspection helpers. They continue to document fields and
derived properties. `PiPLSDataset` keeps its constructor visible because direct user construction
is a supported use case.

## Consequences

- Public result records contain quantities that users inspect or pass downstream, not
  canonicalization bookkeeping or impossible zero blocks.
- The synthetic generator and display-factor helper retain the same numerical behavior.
- Generated reference pages emphasize how returned objects are obtained and what their fields mean.
- Existing pre-release code using the removed fields must use the retained factor arrays or the
  generator's declared structural ranks instead.
- Public-result cleanup API1--API3 is complete.
