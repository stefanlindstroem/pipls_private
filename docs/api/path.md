# Π-PLS path selection

Use `PiPLSSearchCV` to evaluate admissible `(n_components, predictor_rank)` pairs by cross-validation.
`n_components` counts paired latent modes $h$; `predictor_rank` is the retained predictor-subspace
dimension $r_\pi$. The [synthetic tutorial](../tutorials/synthetic.md#retrieve-selection-evidence)
shows the ordinary manual workflow: inspect the search evidence, create one immutable selection,
and pass that exact selection to final refitting.

Every candidate is a cloned `PiPLSRegression` or supported pipeline ending in one. Learned
preprocessing is fitted independently inside each training fold. Before candidate evaluation, the
search object caps the path by the minimum predictor rank verified across those transformed folds.
`PiPLSSearchCV()` is a path evaluator rather than a fitted prediction model. Explicit post-search
`search.select(...)` returns one immutable stored component-path row without fitting.
`search.refit(X, Y, selection=...)` validates an existing selection, clones the configured
estimator or pipeline, fits that clone, attaches the exact supplied object as `model.selection_`,
and returns the model. Rule-based and component-count refitting remain available for compact
workflows.
`search.oof_report(X, Y, selection=...)` consumes an existing selection and produces ordered OOF
diagnostics from the exact validation splits materialized by `fit()`. The search object does not
delegate model methods or retain the returned estimator, report, or supplied training matrices.

For nondefault component requests, predictor-rank policies, rank ceilings, splitters, OOF reporting,
tie-breaking, pipelines, and detailed result surfaces, see
[Path-selection details](../path_analysis.md). For training-cost controls and their statistical or
numerical trade-offs, see [Computational performance](../computational_performance.md). For
candidate-feasibility, refit, scoring, or metadata problems, see
[Troubleshooting](../troubleshooting.md).

`cv_results_` is the complete candidate-level record. Its parameter columns are the stable,
pipeline-independent `n_components` and `predictor_rank` arrays; the remaining columns contain
candidate scores, split values, response-standardized MSE diagnostics, ranks, and timings.
`component_path_` and `predictor_rank_profile()` provide concise immutable views. Use
`search.select(rule="best_score")` for the configured-score optimum on the predictor-rank-conditioned
component path. Post-search `select()`,
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
| Choose a component count after inspecting the path | Create `selection = search.select(n_components=h)` |
| Refit one existing selection | Call `search.refit(X, Y, selection=selection)` |
| Inspect the exact row used by a refitted model | Read `model.selection_` |
| Apply an automatic final rule | Call `search.refit(X, Y, rule=...)` after path evaluation |
| Inspect OOF diagnostics for one selected row | Call `search.oof_report(X, Y, selection=selection)` |

These search-owned operations use the same stored-row vocabulary. Retaining `search` preserves
the complete path and candidate evidence; the returned model owns prediction, transformation,
scoring, and inspection of the final fixed fit.

## Inspect the path and fit one fixed model { #inspect-the-path-and-fit-one-fixed-model }

```python
from pipls import PiPLSSearchCV

CHOSEN_N_COMPONENTS = 2  # application-specific declared choice

search = PiPLSSearchCV().fit(X, Y)
path = search.component_path_
selection = search.select(n_components=CHOSEN_N_COMPONENTS)
rank_profile = search.predictor_rank_profile(selection.n_components)

model = search.refit(X, Y, selection=selection)
```

This example creates the immutable row while inspecting the search evidence, then passes that same
object to final refitting. The returned model exposes the exact supplied object through
`selection_`. Rule-based or component-count refitting remains available when the caller does not
need to retain an earlier selection. The
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
settings such as `scale`, `scale_x`, `scale_y`, `copy`, `svd_solver`, and `random_state`. With
`estimator=None`, the search
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
paired-mode count. With an optimized predictor-rank policy, the search applies its constructor-level
relative and absolute score tolerances independently at every component count and retains the
smallest evaluated qualifying rank. Its aligned read-only arrays support complete path plots and
comparisons without requiring manual masking of `cv_results_`. The predictor-rank policy and
validation split count are stored once as path-wide scalars; row-aligned
`predictor_rank_evidence` records the exact score optimum and resolved tolerances. Use
`search.select(...)` when a
complete scalar row is needed for annotation or reporting. See
[Search-owned selection rules](../path_analysis.md#search-owned-selection-rules) for the rule
definitions and scope. Maintained plots use `cv_mse_std` directly as descriptive split-to-split variability. The path
object provides no standard-error property and no public row-selection methods.

::: pipls.component_path.PiPLSComponentPath
    options:
      show_signature: false

## One selection

`search.select(n_components=h)` returns the frozen scalar row for one evaluated paired-mode count,
including its conditionally selected predictor rank, score, CV-MSE summary, policy, split count, and
`predictor_rank_evidence` when rank was optimized. Direct component-count lookup has `rule is None`.
Named rules record their rule on the result.
A `"minimum_cv_mse"` selection accepts simultaneous relative and absolute tolerances, retains the
exact unruled minimum path row as `reference_minimum`, stores the resolved tolerances, and derives
`cv_mse_threshold`. The default relative tolerance is `sqrt(float64 epsilon)` and the default
absolute tolerance is positive infinity. The threshold is derived from the retained provenance rather
than stored independently.

::: pipls.component_path.PiPLSSelection
    options:
      show_signature: false

## Predictor-rank profile

`predictor_rank_profile(h)` contains every predictor rank actually evaluated for one paired-mode
count, sorted by rank. Under adaptive search this may be a strict subset of the admissible ranks.
`reference_selection` gives the exact configured-score optimum under the private numerical tie rule;
`selection` gives the smallest evaluated rank satisfying the public predictor-rank tolerances and
matches `search.select(n_components=h)`.

::: pipls.component_path.PiPLSPredictorRankProfile
    options:
      show_signature: false

## Predictor-rank evidence

`PiPLSPredictorRankEvidence` stores the exact reference rank, its configured score and CV-MSE
summary, and the resolved relative and absolute tolerances. `score_threshold` is derived in
configured-score units. Fixed and maximum predictor-rank policies have no such evidence.

The Tobacco example uses this evidence to present a scorer-specific CV-MSE threshold while keeping
the API scorer-neutral:

```python
search = PiPLSSearchCV(
    predictor_rank_relative_tolerance=0.10,
).fit(X, Y)
selection = search.select(
    rule="minimum_cv_mse",
    relative_tolerance=0.10,
)
profile = search.predictor_rank_profile(selection.n_components)
model = search.refit(X, Y, selection=selection)
evidence = profile.predictor_rank_evidence
if evidence is None:
    raise RuntimeError("Predictor-rank evidence is unavailable.")
predictor_rank_cv_mse_threshold = -evidence.score_threshold
```

The constructor tolerance controls conditional predictor-rank retention. The `refit()` tolerance is
a separate component-count decision on the already conditioned path.

::: pipls.component_path.PiPLSPredictorRankEvidence
    options:
      show_signature: false

## Out-of-fold report

`search.oof_report(X, Y, selection=...)` fits one existing selection independently on every training
fold from the exact split set materialized by `search.fit()`. The same immutable object can configure
both OOF reporting and final refitting:

```python
selection = search.select(rule="minimum_cv_mse")
report = search.oof_report(X, Y, selection=selection)
model = search.refit(X, Y, selection=selection)
```

The immutable `PiPLSOOFReport` contains the exact supplied `selection`, ordered OOF predictions,
repeated-prediction counts, partial-coverage NaNs, and pooled OOF $R^2$ when at least two rows have
coverage. The supplied selection is validated exactly against the fitted
search, preventing a report for an unrelated component-count or predictor-rank decision. The
operation does not rescore candidates, perform a full-data fit, mutate the search, or retain the
supplied matrices. Each repeated call performs the selected-pair fold fits again; retain and reuse
one report when several tables or figures need the same diagnostics. See
[Computational performance](../computational_performance.md#avoid-repeated-oof-computation).

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
scorer callable. This scoring normalization is independent of the fitted model's `scale_y` policy;
changing response scaling inside `PiPLSRegression` does not disable fold-local response
standardization in this loss.

::: pipls.metrics.response_standardized_mse
    options:
      members: false

::: pipls.metrics.neg_response_standardized_mse
    options:
      members: false
