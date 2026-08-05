# Decision 0145: final implementation-surface cleanup

## Status

Accepted and implemented. All four patches are complete: the guide-layer contract is established,
redundant fitted attributes are removed, model-selection internals and private helpers are
simplified, remaining public modules declare exact exports, and selection ownership is stated
positively. Decision 0151 refines the model-producing workflow to create one selection before OOF
reporting and final refitting.

## Context

A prior pre-release cleanup removed duplicated result access, inconsistent selection terminology,
redundant candidate parameter representations, dataset aliases, unused shape-only inspection
properties, and top-level result re-exports. A follow-up implementation audit found a smaller set
of residual
surface and internal inconsistencies:

- `PiPLSSearchCV.cv_n_train_min_` stores an intermediate fold-size quantity whose only public
  consequence is the learned `max_predictor_rank_` limit;
- `PiPLSRegression.predictor_rank_` duplicates the constructor parameter while no corresponding
  fitted copy of `n_components` exists;
- `pipls.model_selection` has a public-looking module name even though its functions are private
  implementation algorithms;
- `PiPLSCoreResult.predict()` is unused by production code and only repeats multiplication by the
  stored standardized regression map;
- the private OOF engine accepts an `operation_name` argument even though `oof_report()` is now its
  sole caller;
- `pipls.regression`, `pipls.search`, and `pipls.exceptions` lack explicit module-level export
  declarations; and
- two tutorials describe `search.select()` negatively rather than stating its fitting-free role.

The audit found no dead public plotting or numerical inspection functionality. All five inspection
functions remain used by maintained examples, and `search.select()` remains the only way to obtain a
selection without fitting a final model.

## Decision

### Remove redundant fitted attributes

Remove:

```text
PiPLSSearchCV.cv_n_train_min_
PiPLSRegression.predictor_rank_
```

The minimum training-fold size remains an implementation-local input to the learned search limit:

```python
search.max_predictor_rank_
```

A directly configured fixed estimator exposes its configured rank through:

```python
model.predictor_rank
```

A model returned by `search.refit()` exposes the exact fitted pair through selection provenance:

```python
selection = model.selection_
selection.n_components
selection.predictor_rank
```

No fitted underscore copy is retained merely to duplicate an unchanged constructor parameter.
`max_predictor_rank_` remains because it is learned from the fitted data and cross-validation
splits.

### Privatize model-selection algorithms

Rename:

```text
pipls.model_selection
```

to:

```text
pipls._model_selection
```

The module contains implementation algorithms rather than a supported user import surface. Internal
source and focused tests may import the private module. No compatibility module or alias is retained
at version `0.0.0`.

### Remove private helper duplication

Remove `PiPLSCoreResult.predict()`. Core prediction invariants use the stored standardized
regression
map directly:

```python
X @ result.standardized_regression_map
```

Remove the constant `operation_name` argument from the private OOF computation. Its validation
messages refer directly to `oof_report()` because that is the sole public operation using the
engine.

### Declare remaining module export boundaries

Add exact module-level exports:

```python
# pipls.regression
__all__ = ["PiPLSRegression"]

# pipls.search
__all__ = ["PiPLSSearchCV"]

# pipls.exceptions
__all__ = ["PredictorRankSupportWarning"]
```

These declarations control wildcard imports and generated module boundaries. They do not prevent
advanced users from explicitly importing documented public names from their focused modules.

### State the role of `search.select()` positively

Documentation uses this ownership rule:

- `search.select()` obtains an immutable selection without fitting a final model;
- evidence-retaining workflows pass that selection to OOF reporting and `refit(selection=...)`;
- compact workflows may resolve a rule or component count directly in `refit()`;
- every successfully refitted model still exposes the exact fitted selection as `model.selection_`.

Documentation must not explain `search.select()` only by saying that it is not required.

### Retained functionality

This cleanup does not remove:

- `search.select()`;
- `model.selection_`;
- `max_predictor_rank_`, `n_splits_`, `scorer_`, `search_is_exhaustive_`, `cv_results_`, or
  `component_path_`;
- predictor-rank profiles or OOF reporting;
- any of the five public numerical inspection functions;
- caller-owned plotting;
- model-selection algorithms themselves; or
- direct construction, immutability, and pickle validation of public result records.

## Patch sequence

1. Establish this decision and synchronize the guide layer.
2. Remove `cv_n_train_min_` and `predictor_rank_`, migrate consumers, and protect unchanged fitted
   behavior and numerics — complete.
3. Rename `model_selection.py` to `_model_selection.py`, remove `PiPLSCoreResult.predict()`, and
   remove the constant OOF `operation_name` argument — complete.
4. Add remaining module `__all__` declarations, correct tutorial wording, synchronize public
   documentation and maintainer guidance, run final active-surface audits, and mark the decision
   implemented — complete.

No compatibility aliases or deprecation period are required before the first release.

## Validation

Every implementation patch must pass:

```text
git diff --check
Ruff
mypy
complete pytest suite
strict MkDocs build
Python compilation
```

Patch 2 additionally protects unchanged selections, fitted maps, predictions, clone behavior,
pickling, and learned maximum predictor rank. Patch 3 protects exhaustive and adaptive selection,
tie-breaking, split materialization, core regression-map invariants, OOF outputs, and informative
validation messages. Patch 4 protects exact module `__all__` contracts and clean audits for all
removed names and stale wording.

## Consequences

The fixed estimator and search expose only learned fitted state or distinct provenance. Private
algorithms and helper methods are named and scoped as implementation details. Remaining public
modules declare their intended wildcard surface, and documentation gives each selection access path
a positive, non-overlapping role.
