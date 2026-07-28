# Pi-PLS path selection

Use `PiPLSSearchCV` to evaluate admissible `(n_components, predictor_rank)` pairs by cross-validation.
The [synthetic tutorial](../tutorials/synthetic.md#evaluate-the-component-path) shows the ordinary
sequence: inspect `component_path_`, choose a component count, retrieve its conditional predictor
rank, and fit a separate fixed estimator.

Every candidate is a cloned `PiPLSRegression` or supported pipeline ending in one. Learned
preprocessing is fitted independently inside each training fold. Before candidate evaluation, the
search object caps the path by the minimum predictor rank verified across those transformed folds.
The default constructor is selection-only: `PiPLSSearchCV()` uses `refit=False` and leaves final
fixed-model fitting as an explicit user step. A caller may instead declare `selection_rule` and set
`refit=True` to fit that stored path row on all supplied data. Methods that delegate to a selected
estimator are available only after such a refit.

For nondefault component requests, predictor-rank policies, rank ceilings, splitters, OOF reporting,
tie-breaking, pipelines, and detailed result surfaces, see
[Path-selection details](../path_analysis.md). For candidate-feasibility, refit, scoring, or metadata
problems, see [Troubleshooting](../troubleshooting.md).

`cv_results_` is the complete candidate-level record. `component_path_` and
`predictor_rank_profile()` provide concise immutable views. Standard `best_*` attributes identify
the global configured-score optimum, while `selected_result_` identifies the row chosen by the
declared final rule. `validation_report_` and optional OOF arrays describe that selected row.
Python method signatures use `y` by scikit-learn convention even when the
response is a matrix denoted by $Y$ in equations; see the
[API overview](index.md#mathematical-notation-and-python-names).

::: pipls.PiPLSSearchCV
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
requiring manual masking of `cv_results_`. The predictor-rank policy and validation split count are
stored once as path-wide scalars rather than repeated in every row. It also provides non-mutating methods that return the
stored minimum-CV-MSE row or the conventional 1-SE row as complete `PiPLSComponentResult` objects.
These methods inspect evaluated results only; they do not fit, refit, or change `best_*`. See
[Component-path recommendation methods](../path_analysis.md#result-object-recommendations) for the
rule definitions and scope.

::: pipls.PiPLSComponentPath
    options:
      show_signature: false
      members:
        - cv_mse_standard_error
        - minimum_cv_mse_result
        - one_standard_error_result
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
its `selected` property derives the same conditionally selected scalar values returned by
`component_path_` from the immutable candidate arrays and shared policy and split-count scalars.

::: pipls.PiPLSPredictorRankProfile
    options:
      show_signature: false

## Validation report

`PiPLSSearchCV.validation_report_` composes the immutable `selected_result_` with validation
provenance and, when requested, ordered out-of-fold predictions. Its existing component-count,
predictor-rank, split-count, score, and CV-MSE attributes are read-only views of
`validation_report_.selected_result`.

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
better. `PiPLSSearchCV` uses the stable string
`"neg_response_standardized_mean_squared_error"` by default and resolves it to the public negative
scorer callable.

::: pipls.metrics.response_standardized_mean_squared_error
    options:
      members: false

::: pipls.metrics.neg_response_standardized_mean_squared_error
    options:
      members: false
