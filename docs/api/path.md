# Pi-PLS path selection

Use `PiPLSPathCV` to evaluate admissible `(n_components, predictor_rank)` pairs by cross-validation.
The [synthetic tutorial](../tutorials/synthetic.md#evaluate-the-component-path) shows the ordinary
sequence: inspect `component_path_`, choose a component count, retrieve its conditional predictor
rank, and fit a separate fixed estimator.

Every candidate is a cloned `PiPLSRegression` or supported pipeline ending in one. Learned
preprocessing is fitted independently inside each training fold. Before candidate evaluation, the
selector caps the path by the minimum predictor rank verified across those transformed folds.
Methods that delegate to a selected estimator are available only when `refit=True`.

For nondefault component requests, predictor-rank policies, rank ceilings, splitters, OOF reporting,
tie-breaking, pipelines, and detailed result surfaces, see
[Path-selection details](../path_analysis.md). For candidate-feasibility, refit, scoring, or metadata
problems, see [Troubleshooting](../troubleshooting.md).

`cv_results_` is the complete candidate-level record. `component_path_` and
`predictor_rank_profile()` provide concise immutable views, `validation_report_` owns optional OOF
arrays and coverage counts, and standard `best_*` attributes identify the global selected
candidate.

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

## Validation report

`PiPLSPathCV.validation_report_` summarizes the selected cross-validation result and, when
requested, its ordered out-of-fold predictions.

::: pipls.PiPLSValidationReport
    options:
      show_signature: false
      members:
        - selection_conditioned
        - complete_oof_coverage

## Scoring functions

The public scoring functions standardize each response residual by the corresponding sample
standard deviation learned from the estimator's training responses. The positive function reports
an error; the negative function follows the scikit-learn convention that larger scorer values are
better. `PiPLSPathCV` uses the negative form by default.

::: pipls.metrics.response_standardized_mean_squared_error
    options:
      members: false

::: pipls.metrics.neg_response_standardized_mean_squared_error
    options:
      members: false
