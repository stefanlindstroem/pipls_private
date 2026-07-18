# Decision 0035: Tobacco randomized-SVD auto path

## Status

Superseded by Decision 0036.

## Context

The Tobacco integration contains 347 raw FT-NIR spectra, 1,557 predictors, and 13 chemical
responses. Its former example fitted one fixed eight-component model with predictor rank 16 and
printed training $R^2$. That did not follow the accepted two-stage component-path workflow and
could be read as a validated model recommendation.

Tobacco is also the appropriate real-data example for the package's randomized predictor-SVD
policy. A complete path across all 13 response components is too heavy for a concise repository
example, so the computational boundary must be explicit.

## Decision

The Tobacco example and smoke benchmark evaluate component counts 1 through 8. For each component
count, predictor rank is selected conditionally with `PiPLSPathCV(search_method="auto")`.
Candidate fits use an explicit `PiPLSRegression(svd_solver="randomized", random_state=0)` template.
The path uses `refit=False` and writes the same six-column component-path CSV contract as Pulp and
Sugarcane:

```text
n_components
predictor_rank
predictor_rank_policy
response_standardized_cv_mse_mean
response_standardized_cv_mse_fold_sd
n_splits
```

The example generates a PDF by reading the CSV, exposes a visible user component choice, and fits a
separate fixed `PiPLSRegression` with the recorded predictor rank and the same randomized-SVD
policy. The benchmark stops at the CSV path and does not choose or refit a final model.

The eight-component upper bound is an example boundary, not a claim that eight components is
universally optimal or that all admissible response components have been exhausted. Users may
extend `N_COMPONENTS_VALUES` when their analysis and computing budget require a longer path.

## Consequences

- Tobacco demonstrates both adaptive predictor-rank scanning and explicit randomized predictor SVD
  on a substantial $p>n$ real dataset.
- The canonical CSV always records the numeric predictor rank selected for each component count.
- The example no longer reports training $R^2$ or presents fixed ranks without path evidence.
- No timing, memory, full-versus-randomized comparison, spectral preprocessing, or performance
  threshold is added.
- Pulp and Sugarcane remain ordinary/default path examples; Tobacco is the explicit randomized-SVD
  example.
