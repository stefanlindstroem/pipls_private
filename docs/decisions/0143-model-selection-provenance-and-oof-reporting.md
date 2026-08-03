# Decision 0143: model-selection provenance and OOF reporting

## Status

Accepted; implementation in progress (Patches 1 through 3 complete).

## Context

The current search lifecycle separates path evaluation from final fitting, but maintained workflows
must repeat the selection request across several operations. An automatic workflow commonly calls
`search.select(rule="one_standard_error")`, `search.refit(..., rule="one_standard_error")`, and
`search.validation_report(..., rule="one_standard_error")`. A manual workflow similarly repeats an
explicit component count. The returned model does not record which stored path row configured it.

This repetition exposes implementation mechanics rather than the programming user's task. It also
permits a validation report to be requested for a different parameterization than the fitted model.
The report name is broad even though the method specifically recomputes ordered out-of-fold (OOF)
predictions for one selected component-count and predictor-rank pair on the exact splits
materialized
by the fitted search.

The complete examples also interleave model construction, search inspection, and rendering. The
normal model-producing workflow should complete search and full-data refitting first. Selection
provenance, path evidence, conditional rank evidence, OOF diagnostics, fitted-model inspection, and
rendering are subsequent analysis operations. OOF reporting is optional analysis and is not part of
constructing the final model.

The package remains unreleased at version `0.0.0`. The final vocabulary can therefore replace the
pre-release report API without compatibility aliases.

## Decision

### Refitted models retain their selection

Every estimator or pipeline successfully returned by `PiPLSSearchCV.refit()` exposes:

```python
model.selection_
```

The attribute is the exact immutable `PiPLSComponentResult` used to configure the full-data fit. It
is attached to the outer object returned by `refit()` only after fitting succeeds. A
`PiPLSRegression` fitted directly through `fit()` has no `selection_` attribute because no search
selection occurred.

`refit()` continues to return only the fitted estimator. It does not return a tuple or wrapper and
does not attach the model to search state.

### Selection results carry rule provenance

Extend `PiPLSComponentResult` with:

```python
rule: SelectionRule | None
reference_minimum: PiPLSComponentResult | None
```

and the derived property:

```python
one_standard_error_threshold: float | None
```

The fields have these meanings:

| Selection request | `rule` | `reference_minimum` | 1-SE threshold |
|---|---|---|---|
| `n_components=h` | `None` | `None` | `None` |
| `rule="best_score"` | `"best_score"` | `None` | `None` |
| `rule="minimum_cv_mse"` | `"minimum_cv_mse"` | `None` | `None` |
| `rule="one_standard_error"` | `"one_standard_error"` | minimum-CV-MSE result | derived |

For a one-standard-error selection:

```python
minimum = selection.reference_minimum
threshold = selection.one_standard_error_threshold
```

and the threshold is derived exactly as:

```python
minimum.cv_mse_mean + minimum.cv_mse_standard_error
```

The threshold is not stored independently. The reference minimum is the same immutable result that
`search.select(rule="minimum_cv_mse")` would return. It has no recursively nested reference result.

The public class name remains `PiPLSComponentResult` during this transition. Ordinary workflow code
uses the shorter variable and attribute names `selection` and `model.selection_`; renaming the class
would add migration work without simplifying that code.

### OOF reporting consumes an existing selection

Replace the final public report method with:

```python
def oof_report(
    self,
    X: ArrayLike,
    y: ArrayLike,
    *,
    selection: PiPLSComponentResult,
) -> PiPLSOOFReport:
    ...
```

`oof_report()` accepts neither `rule` nor `n_components`. It uses the exact component count and
predictor rank stored in `selection`, validates that the selection is consistent with the fitted
search, and recomputes ordered OOF predictions on defensive copies of the exact splits materialized
by `fit()`.

The immutable report exposes:

```python
report.selection
report.oof_predictions
report.oof_prediction_counts
report.pooled_oof_r2
report.is_leave_one_out
report.has_complete_oof_coverage
```

Repeated validation predictions are averaged, uncovered rows retain `NaN` predictions and zero
counts, and pooled OOF $R^2$ uses covered rows only. The operation does not rescore candidates,
fit a
full-data model, retain supplied matrices, or mutate the search.

The final report type is `PiPLSOOFReport`. The former `PiPLSValidationReport`,
`validation_report()`, `estimate_kind`, and `is_selection_conditioned` names are removed without
aliases. Documentation explains directly that an OOF report based on a search-owned selection is a
selection-conditioned diagnostic, not an independent nested-CV or external-test estimate.

The primary model-producing call is:

```python
report = search.oof_report(
    X,
    Y,
    selection=model.selection_,
)
```

A selection-only workflow may instead use:

```python
selection = search.select(rule="best_score")
report = search.oof_report(X, Y, selection=selection)
```

This retains `search.select()` as an optional, fitting-free operation rather than a required step in
normal refit workflows.

### Workflow order

The standard model-producing workflow is:

```text
data and declared choices
        ↓
search.fit()
        ↓
search.refit()                 modeling complete
        ↓
model.selection_               analysis begins
search.component_path_
search.predictor_rank_profile(...)
search.oof_report(...)         optional OOF analysis
fitted-model inspection
        ↓
rendering
```

The maintained code pattern is:

```python
search = PiPLSSearchCV(cv=cv).fit(X, Y)

model = search.refit(
    X,
    Y,
    rule="one_standard_error",
)

# Analysis follows modeling.
selection = model.selection_
path = search.component_path_
rank_profile = search.predictor_rank_profile(selection.n_components)
report = search.oof_report(X, Y, selection=selection)
```

Manual selection uses the same order with `n_components=CHOSEN_N_COMPONENTS`. OOF reporting remains
optional and is omitted when an external test set already supplies the relevant prediction
diagnostic.

Examples that deliberately produce no final model retain their specialized roles. The leave-one-out
validation example may obtain a selection through `search.select()` and pass it to `oof_report()`.
The path-comparison example need not create unused final models.

### Method roles

```text
search.select()
    optionally inspect one selection without fitting

search.refit()
    fit one full-data model and retain model.selection_

search.oof_report(selection=...)
    compute OOF diagnostics for an existing selection

search.component_path_
    expose aligned component-count evidence

search.predictor_rank_profile(h)
    expose conditional predictor-rank evidence at one component count
```

### Implementation sequence

Implement this transition in seven reviewable patches:

1. establish this decision and the guide-layer target;
2. enrich `PiPLSComponentResult` with rule, reference-minimum, and derived-threshold provenance;
3. retain the resolved selection as `model.selection_` on successful `refit()` results;
4. add `oof_report(selection=...)` and `PiPLSOOFReport` while retaining the former report API only
   as a temporary internal migration bridge;
5. migrate manual-selection examples, tutorial renderers, tutorials, and their structural tests;
6. migrate the automatic Tobacco workflow, quick start, and validation-only workflow;
7. remove the former report API, normalize all maintained examples and tutorials, complete active-
   surface audits, and mark this decision implemented.

## Implementation status

Patches 1 through 3 are complete. `PiPLSComponentResult` carries validated rule provenance,
one-standard-error selections retain the exact minimum-CV-MSE result, and the threshold is derived
from that immutable reference. Every successful `refit()` result now exposes the exact resolved row
as `model.selection_`; direct estimator fitting remains provenance-free, and pipelines retain the
selection on the returned outer object. The examples, renderers, and served reporting documentation
still expose `validation_report()` and `PiPLSValidationReport` until their assigned patches. Patch 4
is the next increment.

This decision refines Decisions 0137 and 0140. Search continues to own candidate evidence, exact
stored splits, and fitting-free selected-row resolution, but final-model provenance now belongs on
the model returned by `refit()`, and OOF diagnostics consume an existing selection rather than
resolving a repeated rule or component count. It also refines Decisions 0108 and 0109 by assigning
the Tobacco minimum and 1-SE threshold annotations to the enriched selection. Decisions 0042,
0061, and 0128 continue to separate numerical analysis from caller-owned rendering.

## Consequences

- The shortest automatic workflow remains one expression returning a fitted model.
- Manual and automatic model-producing workflows no longer require a separate `search.select()`
  call merely to recover the row used by `refit()`.
- A fitted model records exactly which component-count and predictor-rank pair constructed it.
- Automatic 1-SE plots obtain the selected row, reference minimum, minimum-row standard error, and
  threshold from one immutable selection object.
- OOF reporting cannot silently evaluate a different rule than the fitted model when supplied with
  `model.selection_`.
- `oof_report()` states the operation's numerical purpose more precisely than
  `validation_report()`.
- OOF computation remains explicit and optional because it refits the selected pair across stored
  folds and may allocate large prediction arrays.
- Maintained model-producing workflows complete modeling before path, rank-profile, OOF, fitted-
  model, or rendering analysis.
- The pre-release report names are removed rather than deprecated or aliased.
