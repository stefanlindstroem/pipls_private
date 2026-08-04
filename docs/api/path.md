# Pi-PLS path selection

Use `PiPLSSearchCV` to evaluate admissible `(n_components, predictor_rank)` pairs by cross-validation.
`n_components` counts paired latent modes $h$; `predictor_rank` is the retained predictor-subspace
dimension $r_\pi$. The [synthetic tutorial](../tutorials/synthetic.md#retrieve-selection-evidence)
shows the ordinary manual workflow: declare a component count, complete search and refitting, then
inspect `model.selection_`, `component_path_`, and the conditional predictor-rank profile.

Every candidate is a cloned `PiPLSRegression` or supported pipeline ending in one. Learned
preprocessing is fitted independently inside each training fold. Before candidate evaluation, the
search object caps the path by the minimum predictor rank verified across those transformed folds.
`PiPLSSearchCV()` is a path evaluator rather than a fitted prediction model. Explicit post-fit
`search.select(...)` returns one immutable stored component-path row without fitting.
`search.refit(X, Y, ...)` resolves the same row, clones the configured estimator or pipeline, fits
that clone, attaches the exact immutable row as `model.selection_`, and returns the model.
`search.oof_report(X, Y, selection=...)` consumes an existing selection and produces ordered OOF
diagnostics from the exact validation splits materialized by `fit()`. The search object does not
delegate model methods or retain the returned estimator, report, or supplied training matrices.

For nondefault component requests, predictor-rank policies, rank ceilings, splitters, OOF reporting,
tie-breaking, pipelines, and detailed result surfaces, see
[Path-selection details](../path_analysis.md). For candidate-feasibility, refit, scoring, or metadata
problems, see [Troubleshooting](../troubleshooting.md).

`cv_results_` is the complete candidate-level record. Its parameter columns are the stable,
pipeline-independent `n_components` and `predictor_rank` arrays; the remaining columns contain
candidate scores, split values, response-standardized MSE diagnostics, ranks, and timings.
`component_path_` and `predictor_rank_profile()` provide concise immutable views. Use
`search.select(rule="best_score")` for the global configured-score optimum. Post-fit `select()`,
`refit()`, and `oof_report()` do not alter search state. No final selection, model, or OOF report is
stored on the search object.
Python method signatures use `y` by scikit-learn convention even when the
response is a matrix denoted by $\mathbf{Y}$ in equations; see the
[API overview](index.md#mathematical-notation-and-python-names).

## Choose the workflow { #choose-the-workflow }

| Situation | Workflow |
|---|---|
| Both ranks are already known | Fit `PiPLSRegression` directly |
| Inspect one selected row without fitting | Call `search.select(rule=... or n_components=h)` |
| Choose a component count after inspecting the path | Call `search.refit(X, Y, n_components=h)` |
| Inspect the exact row used by a refitted model | Read `model.selection_` |
| Apply an automatic final rule | Call `search.refit(X, Y, rule=...)` after path evaluation |
| Inspect OOF diagnostics for one selected row | Call `search.oof_report(X, Y, selection=selection)` |

These search-owned operations use the same stored-row vocabulary. Retaining `search` preserves
the complete path and candidate evidence; the returned model owns prediction, transformation,
scoring, and inspection of the final fixed fit.

## Fit one fixed model and inspect the path { #inspect-the-path-and-fit-one-fixed-model }

```python
from pipls import PiPLSSearchCV

CHOSEN_N_COMPONENTS = 2  # application-specific declared choice

search = PiPLSSearchCV().fit(X, Y)
model = search.refit(
    X,
    Y,
    n_components=CHOSEN_N_COMPONENTS,
)

# Analysis follows completed modeling.
selection = model.selection_
path = search.component_path_
rank_profile = search.predictor_rank_profile(selection.n_components)
```

This example keeps model construction together and performs numerical analysis afterward. A
refitted model exposes the exact row it used through `selection_`. `select()` remains optional for
selection-only workflows that do not construct a final model. The
[synthetic tutorial](../tutorials/synthetic.md#retrieve-selection-evidence) explains how retained
path evidence can be used to justify an application-specific declared choice.

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
        - oof_report
        - predictor_rank_profile

## Concise component path

`component_path_` contains one conditionally chosen predictor-rank result for each evaluated
paired-mode count. Its aligned read-only arrays support complete path plots and comparisons without
requiring manual masking of `cv_results_`. The predictor-rank policy and validation split count are
stored once as path-wide scalars rather than repeated in every row. Use `search.select(...)` when a
complete scalar row is needed for annotation or reporting. See
[Search-owned selection rules](../path_analysis.md#search-owned-selection-rules) for the rule
definitions and scope. The path object itself provides no public row-selection methods.

::: pipls.component_path.PiPLSComponentPath
    options:
      show_signature: false
      members:
        - cv_mse_standard_error

## One selection

`search.select(n_components=h)` returns the frozen scalar row for one evaluated paired-mode count,
including its conditionally selected predictor rank, score, CV-MSE summary, policy, and split count.
Direct component-count lookup has `rule is None`. Named rules record their rule on the result.
A `"minimum_cv_mse"` selection accepts simultaneous relative and absolute tolerances, retains the
exact unruled minimum path row as `reference_minimum`, stores the resolved tolerances, and derives
`cv_mse_threshold`. The default relative tolerance is `sqrt(float64 epsilon)` and the default
absolute tolerance is positive infinity. A temporary `"one_standard_error"` result also references
the exact unruled minimum row and derives `one_standard_error_threshold`. Neither threshold is
stored independently.

::: pipls.component_path.PiPLSSelection
    options:
      show_signature: false

## Predictor-rank profile

`predictor_rank_profile(h)` contains every predictor rank actually evaluated for one paired-mode
count, sorted by rank. Under adaptive search this may be a strict subset of the admissible ranks;
its `selection` property derives the same conditional selection returned by
`search.select(n_components=h)` from the immutable candidate arrays and shared policy and
split-count scalars.

::: pipls.component_path.PiPLSPredictorRankProfile
    options:
      show_signature: false

## Out-of-fold report

`search.oof_report(X, Y, selection=...)` fits one existing selection independently on every training
fold from the exact split set materialized by `search.fit()`. A model returned by `search.refit(...)`
exposes the intended value as `model.selection_`:

```python
selection = model.selection_
report = search.oof_report(X, Y, selection=selection)
```

The immutable `PiPLSOOFReport` contains the exact supplied `selection`, ordered OOF predictions,
repeated-prediction counts, partial-coverage NaNs, leave-one-out provenance, and pooled OOF $R^2$
when at least two rows have coverage. The supplied selection is validated exactly against the fitted
search, preventing a report for an unrelated component-count or predictor-rank decision. The
operation does not rescore candidates, perform a full-data fit, mutate the search, or retain the
supplied matrices.

The caller must provide the same observations in the same row order and with the same sample and
response-column counts as the fitted search. Selection metrics remain on `report.selection`;
`has_complete_oof_coverage` summarizes row coverage. Because the search splits also produced the
selection, the report is a
selection-conditioned diagnostic rather than an independent performance estimate.

::: pipls.validation.PiPLSOOFReport
    options:
      show_signature: false
      members:
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
