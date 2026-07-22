# Pi-PLS path selection

Use `PiPLSPathCV` to evaluate admissible `(n_components, predictor_rank)` pairs by cross-validation.
The [Pulp tutorial](../tutorials/pulp.md#evaluate-the-component-path) shows the ordinary sequence:
inspect `component_path_`, choose a component count, retrieve its conditional predictor rank, and
fit a separate fixed estimator.

Every candidate is a cloned `PiPLSRegression` or supported pipeline ending in one. Learned
preprocessing is therefore fitted independently inside each training fold. Methods that delegate to
a selected estimator are available only when `refit=True`.

For nondefault component requests, predictor-rank policies, rank ceilings, tie-breaking, pipelines,
and detailed result surfaces, see [Advanced path-search behavior](../path_analysis.md).

::: pipls.PiPLSPathCV
    options:
      members:
        - fit
        - predictor_rank_profile
        - predict
        - transform
        - fit_transform
        - inverse_transform
        - score
        - get_feature_names_out

## Concise component path

::: pipls.PiPLSComponentPath
    options:
      members:
        - for_n_components

## One component result

::: pipls.PiPLSComponentResult

## Predictor-rank profile

::: pipls.PiPLSPredictorRankProfile
