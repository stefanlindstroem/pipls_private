# Decision 0141: spectral predictor-rank profile figures

## Status

Accepted and implemented. Decision 0146 refines the Tobacco component-count rule and replaces
standard-error presentation with split-SD presentation. Decision 0148 adds the separate predictor-
rank tolerance demonstration. Decision 0151 refines the workflow so the profile and final refit
consume one selection created before fitting the final model.

## Context

The complete Pulp workflow demonstrates the second search axis: after selecting a component count,
it calls `predictor_rank_profile()` to inspect every predictor rank evaluated at that count.
Sugarcane and Tobacco should expose the same conditional evidence rather than present the selected
predictor rank only as one scalar.

The selected component count may be manual or rule-based. The profile must therefore consume the
component count retained by the exact selection that also configures the final model rather than
duplicate it as a literal or reconstruct candidate rows from `cv_results_`.

## Decision

Examples 05 and 06 use the public composition:

```python
selection = search.select(...)
rank_profile = search.predictor_rank_profile(selection.n_components)
model = search.refit(X, Y, selection=selection)
```

Sugarcane retains its declared component count. Tobacco obtains its component count from
`minimum_cv_mse` with a 10% relative tolerance, as defined by Decision 0146. Decision 0148 also
configures a separate 10% predictor-rank relative tolerance on the Tobacco search. Neither example
repeats the selected count as a separate rank-profile input.

Each example owns a private same-file renderer that plots:

- evaluated `predictor_rank` values;
- aligned `cv_mse_mean` values with `cv_mse_std` error bars;
- the exact conditional reference exposed by `rank_profile.reference_selection`;
- the tolerance-qualified retained row exposed by `rank_profile.selection`;
- the predictor-rank tolerance threshold when the two rows differ.

The SD bars describe split-to-split dispersion. They are not confidence intervals. Under the
maintained default scorer, the exact reference is the conditional CV-MSE minimum and the retained
rank is the smallest evaluated rank satisfying both predictor-rank tolerance caps.

Both examples write `predictor_rank_profile.pdf` in addition to their other outputs. Rendering
remains caller-owned and consumes immutable public result arrays. No plotting API or additional
fitted state is added.

## Consequences

- All three complete real-data workflows expose component-path and conditional predictor-rank
  evidence.
- Tobacco demonstrates two explicit sequential parsimony decisions: tolerance-based predictor-
  rank retention followed by tolerance-based component-count selection.
- Adaptive-search omissions remain visible because profiles contain only evaluated ranks.
- Sugarcane and Tobacco retain compact `main()` orchestration and same-file rendering.
- The package continues to expose no public plotting API.
