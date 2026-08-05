# Decision 0072: conditional predictor-rank profile

## Status

Accepted and implemented. Public result naming and selection ownership are refined by Decisions
0140, 0145, and 0146.

## Context

The concise component path stores one conditionally selected predictor rank per component count.
Scientific inspection sometimes requires every predictor rank actually evaluated at one fixed
component count, especially for spectral examples. Duplicating all candidate arrays in every path
row would make the concise path large and create synchronization risk with `cv_results_`.

## Decision

A fitted search exposes:

```python
profile = search.predictor_rank_profile(n_components)
```

The returned immutable `PiPLSPredictorRankProfile` is derived on demand from `cv_results_` and
contains:

```text
n_components
predictor_rank
mean_test_score
cv_mse_mean
cv_mse_std
predictor_rank_policy
n_splits
selection
```

Candidate arrays contain only ranks actually evaluated for the requested component count and are
sorted in strictly ascending predictor-rank order. The derived `selection` is a `PiPLSSelection`
using the same configured-score optimum and tolerant tie-breaking rule as the fitted search. Under
the default scorer this is equivalent to the minimum mean response-standardized CV-MSE candidate
at that component count.

The method requires a fitted search and an evaluated integer component count. It does not cache a
mapping of every possible profile, mutate the search, fit a model, or select a component count.

## Consequences

- The component path remains concise.
- `cv_results_` remains the single candidate-level source of truth.
- Spectral examples can show rank-wise mean CV-MSE and split SD at the fitted component count.
- Profile inspection and path selection remain separate operations with separate result types.
