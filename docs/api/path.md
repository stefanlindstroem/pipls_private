# Pi-PLS path selection

Use `PiPLSSearchCV` to evaluate admissible `(n_components, predictor_rank)` pairs by cross-validation.
`n_components` counts paired latent modes $h$; `predictor_rank` is the retained predictor-subspace
dimension $r_\pi$. The [synthetic tutorial](../tutorials/synthetic.md#evaluate-the-component-path)
shows the ordinary sequence: inspect `component_path_`, choose a paired-mode count, inspect the
complete stored row with `search.select(...)`, and fit it with `search.refit(...)`.

Every candidate is a cloned `PiPLSRegression` or supported pipeline ending in one. Learned
preprocessing is fitted independently inside each training fold. Before candidate evaluation, the
search object caps the path by the minimum predictor rank verified across those transformed folds.
`PiPLSSearchCV()` is a path evaluator rather than a fitted prediction model. Explicit post-fit
`search.select(...)` returns one immutable stored component-path row without fitting.
`search.refit(X, Y, ...)` resolves the same row, clones the configured estimator or pipeline, fits
that clone, and returns it. `search.validation_report(X, Y, ...)` selects a row through the same
rules and produces ordered OOF
diagnostics from the exact validation splits materialized by `fit()`. The search object does not
delegate model methods or retain the returned estimator or supplied training matrices.

For nondefault component requests, predictor-rank policies, rank ceilings, splitters, OOF reporting,
tie-breaking, pipelines, and detailed result surfaces, see
[Path-selection details](../path_analysis.md). For candidate-feasibility, refit, scoring, or metadata
problems, see [Troubleshooting](../troubleshooting.md).

`cv_results_` is the complete candidate-level record. `component_path_` and
`predictor_rank_profile()` provide concise immutable views. Standard `best_*` attributes identify
the global configured-score optimum. Post-fit `select()`, `refit()`, and `validation_report()` use
the same selection vocabulary and do not alter search state. No final selection or validation
report is stored on the search object.
Python method signatures use `y` by scikit-learn convention even when the
response is a matrix denoted by $\mathbf{Y}$ in equations; see the
[API overview](index.md#mathematical-notation-and-python-names).

## Choose the workflow { #choose-the-workflow }

| Situation | Workflow |
|---|---|
| Both ranks are already known | Fit `PiPLSRegression` directly |
| Inspect one selected row without fitting | Call `search.select(rule=... or n_components=h)` |
| Choose a component count after inspecting the path | Call `search.refit(X, Y, n_components=h)` |
| Apply an automatic final rule | Call `search.refit(X, Y, rule=...)` after path evaluation |
| Inspect OOF diagnostics for one selected row | Call `search.validation_report(X, Y, rule=... or n_components=h)` |

All three search-owned operations use the same stored-row vocabulary. Retaining `search` preserves
the complete path and candidate evidence; the returned model owns prediction, transformation,
scoring, and inspection of the final fixed fit.

## Inspect the path and fit one fixed model { #inspect-the-path-and-fit-one-fixed-model }

```python
from pipls import PiPLSSearchCV

search = PiPLSSearchCV().fit(X, Y)
path = search.component_path_

CHOSEN_N_COMPONENTS = 2  # application-specific choice after inspecting the path
selected = search.select(n_components=CHOSEN_N_COMPONENTS)  # optional scalar evidence

model = search.refit(
    X,
    Y,
    n_components=CHOSEN_N_COMPONENTS,
)
```

This example shows the executable selection-to-fit contract. `select()` is optional and useful for
annotations or reports; `refit()` resolves the same stored row internally. The
[synthetic tutorial](../tutorials/synthetic.md#evaluate-the-component-path) explains how to inspect
and interpret the component path before making the application-specific choice.

## Configure the candidate estimator { #configure-the-candidate-estimator }

Settings that control candidate fitting belong to the supplied estimator template, not to separate
`PiPLSSearchCV` parameters:

```python
from pipls import PiPLSRegression, PiPLSSearchCV

template = PiPLSRegression(
    n_components=1,
    predictor_rank=1,
    scale=True,
    svd_solver="full",
    random_state=0,
)

search = PiPLSSearchCV(estimator=template).fit(X, Y)
model = search.refit(
    X,
    Y,
    n_components=CHOSEN_N_COMPONENTS,
)
```

The pair `(1, 1)` is only a valid construction seed. The search replaces `n_components` and
`predictor_rank` for fold-rank preflight and candidate fitting; cloning preserves other template
settings such as `scale`, `copy`, `svd_solver`, and `random_state`. With `estimator=None`, the search
creates the same seed pair using the ordinary `PiPLSRegression` defaults. A pipeline is configured
in the same way through its terminal `PiPLSRegression` step; see
[Pipelines and fold-local preprocessing](../path_analysis.md#pipelines-and-fold-local-preprocessing).

`svd_solver` controls only the initial predictor-matrix SVD. The response cross-product and latent
coupling SVDs remain exact. Inspect the solver actually used on the returned model:

```python
search = PiPLSSearchCV(estimator=template).fit(X, Y)
model = search.refit(X, Y, n_components=CHOSEN_N_COMPONENTS)
model.decomposition_.predictor_svd_solver
```

When the configured template is a pipeline, inspect its fitted terminal `PiPLSRegression` step.

::: pipls.PiPLSSearchCV
    options:
      members:
        - fit
        - select
        - refit
        - validation_report
        - predictor_rank_profile

## Concise component path

`component_path_` contains one conditionally selected predictor-rank result for each evaluated
paired-mode count. Its aligned read-only arrays support complete path plots and comparisons without
requiring manual masking of `cv_results_`. The predictor-rank policy and validation split count are
stored once as path-wide scalars rather than repeated in every row. During the staged transition,
the path still provides non-mutating row lookup and recommendation methods; new code should use
`search.select(...)`. The transitional methods inspect evaluated results only; they do not fit,
refit, or change `best_*`. See
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

`search.select(n_components=h)` returns the frozen scalar row for one evaluated paired-mode count,
including its conditionally selected predictor rank, score, CV-MSE summary, policy, and split count.
The path-level lookup remains temporarily for migration parity and will be removed before release.

::: pipls.PiPLSComponentResult
    options:
      show_signature: false

## Predictor-rank profile

`predictor_rank_profile(h)` contains every predictor rank actually evaluated for one paired-mode
count, sorted by rank. Under adaptive search this may be a strict subset of the admissible ranks;
its `selected_result` property derives the same conditionally selected scalar values returned by
`search.select(n_components=h)` from the immutable candidate arrays and shared policy and
split-count scalars.

::: pipls.PiPLSPredictorRankProfile
    options:
      show_signature: false

## Validation report

`search.validation_report(X, Y, rule=... or n_components=...)` selects one stored component-path row
and fits that fixed parameterization independently on every training fold from the exact split set
materialized by `search.fit()`. It returns an immutable `PiPLSValidationReport` with ordered OOF
predictions, repeated-prediction counts, partial-coverage NaNs, leave-one-out provenance, and pooled
OOF $R^2$ when at least two rows have coverage. The operation does not rescore candidates, perform a
full-data fit, mutate the search, or retain the supplied matrices.

The caller must provide the same observations in the same row order and with the same sample and
response-column counts as the fitted search. The report's `n_components`, `predictor_rank`,
`n_splits`, `mean_test_score`, and `cv_mse_mean` properties are read-only views of its immutable
`selected_result`; `is_selection_conditioned` and `has_complete_oof_coverage` expose provenance and
coverage as predicates.

::: pipls.PiPLSValidationReport
    options:
      show_signature: false
      members:
        - is_selection_conditioned
        - has_complete_oof_coverage

## Scoring functions

The public scoring functions standardize each response residual by the corresponding sample
standard deviation learned from the estimator's training responses. The positive function reports
an error; the negative function follows the scikit-learn convention that larger scorer values are
better. `PiPLSSearchCV` uses the stable string
`"neg_response_standardized_mse"` by default and resolves it to the public negative
scorer callable.

::: pipls.metrics.response_standardized_mse
    options:
      members: false

::: pipls.metrics.neg_response_standardized_mse
    options:
      members: false
