# Pi-PLS path selection

Use `PiPLSPathCV` to evaluate admissible `(n_components, predictor_rank)` pairs by cross-validation.
The [synthetic tutorial](../tutorials/synthetic.md#evaluate-the-component-path) shows the ordinary
sequence: inspect `component_path_`, choose a component count, retrieve its conditional predictor
rank, and fit a separate fixed estimator.

Every candidate is a cloned `PiPLSRegression` or supported pipeline ending in one. Learned
preprocessing is fitted independently inside each training fold. Before candidate evaluation, the
selector caps the path by the minimum predictor rank verified across those transformed folds.
The default constructor is selection-only: `PiPLSPathCV()` uses `refit=False` and leaves final
fixed-model fitting as an explicit user step. Methods that delegate to a selected estimator are
available only when `refit=True`.

For nondefault component requests, predictor-rank policies, rank ceilings, splitters, OOF reporting,
tie-breaking, pipelines, and detailed result surfaces, see
[Path-selection details](../path_analysis.md). For candidate-feasibility, refit, scoring, or metadata
problems, see [Troubleshooting](../troubleshooting.md).

`cv_results_` is the complete candidate-level record. `component_path_` and
`predictor_rank_profile()` provide concise immutable views, `validation_report_` owns optional OOF
arrays and coverage counts, and standard `best_*` attributes identify the global selected
candidate. Python method signatures use `y` by scikit-learn convention even when the
response is a matrix denoted by $Y$ in equations; see the
[API overview](index.md#mathematical-notation-and-python-names).

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

`component_path_` contains one conditionally selected predictor-rank result for each evaluated
component count. Its aligned read-only arrays support complete path plots and comparisons without
requiring manual masking of `cv_results_`.

::: pipls.PiPLSComponentPath
    options:
      show_signature: false
      members:
        - cv_mse_standard_error
        - for_n_components

## One component result

`component_path_.for_n_components(h)` returns the frozen scalar row for one evaluated component
count, including its conditionally selected predictor rank, score, CV-MSE summary, policy, and
split count.

::: pipls.PiPLSComponentResult
    options:
      show_signature: false

## Predictor-rank profile

`predictor_rank_profile(h)` contains every predictor rank actually evaluated for one component
count, sorted by rank. Under adaptive search this may be a strict subset of the admissible ranks;
its `selected` field is the same scalar result returned by `component_path_`.

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
better. `PiPLSPathCV` uses the stable string
`"neg_response_standardized_mean_squared_error"` by default and resolves it to the public negative
scorer callable.

::: pipls.metrics.response_standardized_mean_squared_error
    options:
      members: false

::: pipls.metrics.neg_response_standardized_mean_squared_error
    options:
      members: false
