# Decision 0141: spectral predictor-rank profile figures

## Status

Accepted and implemented.

## Context

The complete Pulp workflow already demonstrates the second selection axis exposed by a fitted
`PiPLSSearchCV`: after choosing a component count, it calls
`predictor_rank_profile()` to inspect every predictor rank evaluated at that count.

The complete Sugarcane and Tobacco workflows inspect only the component path. This leaves the
conditionally selected predictor rank visible only as one scalar on the selected row, even though
the fitted search already retains the aligned candidate evidence needed to explain that rank.

The omission is most important in the Tobacco workflow. Its component count is not a fixed input:
`search.select(rule="one_standard_error")` chooses the smallest eligible component count. A useful
example should therefore show how the returned count becomes the input to the conditional
predictor-rank profile without manual `cv_results_` filtering or parameter transfer.

## Decision

Examples 06 and 07 derive the immutable profile through the same public pattern:

```python
selected = search.select(...)
rank_profile = search.predictor_rank_profile(selected.n_components)
```

Sugarcane continues to choose its component count explicitly. Tobacco continues to obtain
`selected` through the `"one_standard_error"` rule, then requests the predictor-rank profile at that
returned component count. The Tobacco workflow must not duplicate the selected count as a literal or
reconstruct the profile from `cv_results_`.

Each example adds a private same-file rendering function that plots:

- the evaluated `predictor_rank` values;
- aligned `cv_mse_mean` values with `cv_mse_standard_error`;
- the conditionally selected row exposed by `rank_profile.selected_result`.

Under the maintained default scorer, the selected rank is the conditional CV-MSE minimum. The
Tobacco figure title identifies that its component count came from the 1-SE choice so the two-stage
selection logic remains explicit.

Both examples write `predictor_rank_profile.pdf` in addition to their existing outputs. Sugarcane
and Tobacco therefore each write six final PDF files. The new rendering remains caller-owned and
uses only immutable public result arrays. No package API, search numerical behavior, fitted state, or
shared plotting utility changes.

Structural tests require both spectral examples to call
`search.predictor_rank_profile(selected.n_components)` and to retain the six-file artifact
inventory.

This decision refines Decisions 0067, 0069, 0072, 0108, 0128, and 0140. It supersedes only the
five-file Sugarcane and Tobacco artifact inventory recorded by Decision 0128.

## Consequences

- All three complete Pi-PLS real-data workflows expose both component-path and conditional
  predictor-rank evidence.
- Tobacco demonstrates composition between a named component-selection rule and conditional
  predictor-rank inspection.
- Adaptive-search omissions remain visible because the profile contains only ranks actually
  evaluated by the fitted search.
- Sugarcane and Tobacco retain compact `main()` orchestration and private same-file rendering.
- The package continues to expose no plotting API and stores no additional fitted result.
