# Decision 0072: conditional predictor-rank profile

## Status

Accepted and implemented.

## Context

`PiPLSComponentPath` provides one conditionally selected predictor rank for each evaluated component
count. The complete candidate rows remain available through `PiPLSPathCV.cv_results_`.

The Pulp workflow also needs the complete one-dimensional predictor-rank profile at one chosen
component count. Obtaining that profile from `cv_results_` required user code to construct a Boolean
mask, apply the mask independently to several columns, sort the masked rows, and preserve their
alignment. The numbered example, tutorial renderer, and workflow test duplicated this low-level
selection logic.

A stored mapping of profiles would recreate the synchronization problem removed by Decision 0071.
Putting competing candidate arrays inside `PiPLSComponentResult` would also blur the meaning of that
single selected result.

## Decision

Add the public method:

```python
profile = search.predictor_rank_profile(n_components)
```

The method derives the profile on demand from the fitted `cv_results_` dictionary. It returns a
frozen `PiPLSPredictorRankProfile` containing:

```text
n_components
predictor_rank
mean_test_score
cv_mse_mean
cv_mse_fold_sd
n_splits
selected
```

The four numerical arrays are defensive, read-only copies aligned by row. `predictor_rank` is unique
and strictly ascending and contains only candidates actually evaluated by the fitted search.
`selected` is the same `PiPLSComponentResult` returned by
`component_path_.for_n_components(n_components)`.

Conditional selection remains scorer-general: it maximizes the configured mean test score and uses
the fitted lower-rank tie-break. Under the default negative response-standardized-MSE scorer, this
is equivalent to minimizing mean response-standardized CV-MSE. The CV-MSE arrays remain descriptive
when another scorer controls selection.

The method raises the standard scikit-learn not-fitted error before fitting and the same clear
component-count error as `for_n_components()` when the requested count was not evaluated.

Migrate the Pulp example, tutorial renderer, and their tests to this public result. Retain
`cv_results_` as the complete candidate-level source of truth for split scores, timing columns, and
multi-component analyses.

## Consequences

- A standard rank-profile workflow becomes two direct lines rather than manual masking and sorting.
- Adaptive-search omissions remain explicit: the profile includes only evaluated ranks.
- The result is immutable and pickleable, with defensive read-only arrays.
- No additional fitted attribute or duplicate stored search representation is introduced.
- `PiPLSComponentResult` remains a scalar selected row rather than a container for competing rows.
- The Pulp example and tutorial renderer no longer depend on `cv_results_` column names for this
  ordinary inspection task.
