# Decision 0143: model-selection provenance and OOF reporting

## Status

Accepted and implemented. Decision 0146 refines the selection-policy details while this decision
remains canonical for model-owned selection provenance and selection-conditioned OOF reporting.

## Context

Path evaluation, final fitting, and out-of-fold diagnostics are separate operations. Repeating a
selection request independently across those operations can silently fit one parameterization and
report another. A refitted model therefore needs to retain the exact immutable path row that
configured it, and OOF reporting needs to consume that existing selection rather than resolve a
second rule or component count.

The search already owns the materialized validation splits. Reusing those exact splits preserves
the fitted search protocol, including repeated cross-validation and custom splitters, without
creating a second splitter or an independent performance estimate.

## Decision

### Refitted models retain the exact selection

Every estimator or terminal pipeline returned successfully by `PiPLSSearchCV.refit()` exposes:

```python
model.selection_
```

The value is the exact immutable `PiPLSSelection` used to configure the full-data fit. It is
attached to the returned outer object only after fitting succeeds. A `PiPLSRegression` fitted
directly has no `selection_` attribute because no search-owned selection occurred.

`refit()` returns only the fitted estimator. It does not retain a selected fitted model on the
search object and does not return a wrapper or tuple.

### Selection results retain policy provenance

A `PiPLSSelection` records the selected component count, aligned predictor rank, configured score,
CV-MSE summaries, split count, and selection rule. Manual and best-score selections have no
reference minimum or tolerance provenance.

For tolerance-based `minimum_cv_mse` selection, the result also records:

```python
selection.reference_minimum
selection.relative_tolerance
selection.absolute_tolerance
selection.cv_mse_threshold
```

`reference_minimum` is the exact unruled minimum-CV-MSE path row. The threshold is derived from the
stored minimum and resolved tolerances rather than stored independently. Decision 0146 defines the
selection inequality and validation rules.

### OOF reporting consumes an existing selection

The public operation is:

```python
def oof_report(
    self,
    X: ArrayLike,
    y: ArrayLike,
    *,
    selection: PiPLSSelection,
) -> PiPLSOOFReport:
    ...
```

`oof_report()` accepts neither a rule nor a component count. It validates that `selection` is
exactly compatible with the fitted search, then refits that selected pair on defensive copies of
every validation split materialized by `fit()`.

The immutable report exposes:

```python
report.selection
report.oof_predictions
report.oof_prediction_counts
report.pooled_oof_r2
report.is_leave_one_out
report.has_complete_oof_coverage
```

Repeated validation predictions for one observation are averaged. Their multiplicity is retained
in `oof_prediction_counts`. Uncovered rows retain `NaN` predictions and zero counts, and pooled OOF
$R^2$ uses covered rows only. The operation does not rescore candidates, fit a full-data model,
retain supplied matrices, or mutate the search.

An OOF report based on a search-owned selection is a selection-conditioned diagnostic. It is not
nested cross-validation or an external-test estimate.

### Workflow order

The standard model-producing workflow is:

```python
search = PiPLSSearchCV(cv=cv).fit(X, Y)
model = search.refit(X, Y, rule="minimum_cv_mse")

# Analysis follows modeling.
selection = model.selection_
path = search.component_path_
rank_profile = search.predictor_rank_profile(selection.n_components)
report = search.oof_report(X, Y, selection=selection)
```

Manual component-count selection uses the same order with `n_components=...`. A selection-only
workflow may call `search.select(...)` and pass that result to `oof_report()`. OOF reporting remains
optional because it performs one additional fit per stored validation split.

### Method roles

```text
search.select()
    inspect one immutable selection without fitting

search.refit()
    fit one full-data model and retain model.selection_

search.oof_report(selection=...)
    compute OOF diagnostics for an existing compatible selection

search.component_path_
    expose aligned component-count evidence

search.predictor_rank_profile(h)
    expose conditional predictor-rank evidence at one component count
```

## Consequences

- A fitted model records exactly which component-count and predictor-rank pair constructed it.
- OOF reporting cannot silently evaluate a different selection than the fitted model when supplied
  with `model.selection_`.
- Repeated-CV reports inherit every search repetition and expose the resulting prediction counts.
- Search remains the owner of path evidence and materialized splits, but not of a refitted model.
- Selection, OOF analysis, fitted-model inspection, and rendering remain separate operations.
- The removed pre-release validation-report and selected-search-state names have no aliases.
