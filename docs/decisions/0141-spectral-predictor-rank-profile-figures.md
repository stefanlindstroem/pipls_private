# Decision 0141: spectral predictor-rank profile figures

## Status

Accepted and implemented. Decision 0146 refines the Tobacco selection rule and replaces
standard-error presentation with split-SD presentation.

## Context

The complete Pulp workflow demonstrates the second search axis: after selecting a component count,
it calls `predictor_rank_profile()` to inspect every predictor rank evaluated at that count.
Sugarcane and Tobacco should expose the same conditional evidence rather than present the selected
predictor rank only as one scalar.

The selected component count may be manual or rule-based. The profile must therefore consume the
component count retained by the actual fitted-model selection rather than duplicate it as a literal
or reconstruct candidate rows from `cv_results_`.

## Decision

Examples 06 and 07 use the public composition:

```python
selection = model.selection_
rank_profile = search.predictor_rank_profile(selection.n_components)
```

Sugarcane retains its declared component count. Tobacco obtains its component count from
`minimum_cv_mse` with a 10% relative tolerance, as defined by Decision 0146. Neither example repeats
the selected count as a separate rank-profile input.

Each example owns a private same-file renderer that plots:

- evaluated `predictor_rank` values;
- aligned `cv_mse_mean` values with `cv_mse_std` error bars;
- the conditionally selected row exposed by `rank_profile.selection`.

The SD bars describe split-to-split dispersion. They are not confidence intervals or selection
thresholds. Under the maintained default scorer, the selected rank is the conditional CV-MSE
minimum among ranks evaluated at that component count.

Both examples write `predictor_rank_profile.pdf` in addition to their other outputs. Rendering
remains caller-owned and consumes immutable public result arrays. No plotting API or additional
fitted state is added.

## Consequences

- All three complete real-data workflows expose component-path and conditional predictor-rank
  evidence.
- Tobacco demonstrates composition between tolerance-based component selection and conditional
  predictor-rank inspection.
- Adaptive-search omissions remain visible because profiles contain only evaluated ranks.
- Sugarcane and Tobacco retain compact `main()` orchestration and same-file rendering.
- The package continues to expose no public plotting API.
