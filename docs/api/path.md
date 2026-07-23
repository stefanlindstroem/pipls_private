# Pi-PLS path selection

Use `PiPLSPathCV` to evaluate admissible `(n_components, predictor_rank)` pairs by cross-validation.
The [synthetic tutorial](../tutorials/synthetic.md#evaluate-the-component-path) shows the ordinary sequence:
inspect `component_path_`, choose a component count, retrieve its conditional predictor rank, and
fit a separate fixed estimator.

Every candidate is a cloned `PiPLSRegression` or supported pipeline ending in one. Learned
preprocessing is therefore fitted independently inside each training fold. Methods that delegate to
a selected estimator are available only when `refit=True`.

For nondefault component requests, predictor-rank policies, rank ceilings, tie-breaking, pipelines,
and detailed result surfaces, see [Advanced path-search behavior](../path_analysis.md).
For candidate-feasibility, refit, scoring, or metadata problems, see [Troubleshooting](../troubleshooting.md).

The public fitted surface is deliberately compact. `cv_results_` is the complete candidate-level
record; `component_path_` and `predictor_rank_profile()` provide concise immutable views;
`validation_report_` owns optional OOF arrays and coverage counts; and standard `best_*` attributes
identify the global selected candidate. Adaptive-search batch history and duplicate OOF aliases are
not retained as public fitted attributes.

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
      show_signature: false
      members:
        - for_n_components

## One component result

::: pipls.PiPLSComponentResult
    options:
      show_signature: false

## Predictor-rank profile

::: pipls.PiPLSPredictorRankProfile
    options:
      show_signature: false
