# Decision 0144: pre-release public-surface cleanup

## Status

Accepted. Implementation is planned in seven patches; Patch 1 records the target only.

## Context

The package remains unreleased at version `0.0.0`, but several public conveniences now duplicate
information owned by another result or preserve an earlier scikit-learn-like presentation that no
maintained workflow uses. The completed selection-provenance transition made those overlaps more
visible:

- `PiPLSOOFReport` forwards selection fields that are already available through
  `report.selection` and permits absent prediction arrays even though `oof_report()` always
  computes them;
- `PiPLSComponentResult` now represents a complete selection, including rule and 1-SE provenance,
  rather than merely one anonymous component-path row;
- `PiPLSPredictorRankProfile.selected_result` uses different vocabulary for the same selection
  concept exposed by `search.select()`, `model.selection_`, and `report.selection`;
- fitted searches expose five global `best_*` attributes that duplicate
  `search.select(rule="best_score")`;
- `cv_results_` carries three parameter representations for every candidate;
- `PiPLSDataset` exposes both mathematical `X`/`Y` names and scikit-learn-style `data`/`target`
  aliases;
- several inspection records expose shape-only properties that repeat public array shapes; and
- returned result classes are re-exported from top-level `pipls` even though normal users consume
  them through estimator methods and focused submodules.

The same audit found functionality that is not obsolete. `search.select()` is the sole fitting-free
selection operation and is used by the model-free leave-one-out workflow. Every public numerical
inspection function is used by maintained examples. The package has no public plotting module, and
all private plotting functions in maintained examples are called. Direct-construction validation
also protects internal result creation and pickle reconstruction.

The cleanup must therefore remove duplication without erasing distinct capabilities.

## Decision

### Ownership rule

A public quantity is exposed once through the object that owns it:

```text
selection
    selected component count, predictor rank, rule, and 1-SE reference evidence

OOF report
    OOF predictions, counts, coverage, and pooled OOF R2

component path and rank profile
    aligned curve arrays for inspection

cv_results_
    advanced candidate-level scores, folds, and timings

reference dataset
    X, Y, labels, identifiers, provenance, and metadata
```

Convenience accessors that only forward another public object or repeat an array shape are removed
unless they have a demonstrated workflow role.

### Retain fitting-free selection

`PiPLSSearchCV.select()` remains public. Its positive role is:

```python
selection = search.select(rule="best_score")
```

Use it when a selection is needed without fitting a final model. After `refit()`, use
`model.selection_`. Documentation must not describe `select()` only by saying that it is "not
required".

### Simplify OOF reports

`PiPLSOOFReport` retains only:

```python
report.selection
report.oof_predictions
report.oof_prediction_counts
report.pooled_oof_r2
report.is_leave_one_out
report.has_complete_oof_coverage
```

Remove the forwarding properties:

```text
n_components
predictor_rank
n_splits
mean_test_score
cv_mse_mean
```

`oof_predictions` and `oof_prediction_counts` become required arrays. A successfully returned OOF
report cannot represent their absence. `pooled_oof_r2` remains optional because fewer than two rows
may have OOF coverage.

### Use selection terminology consistently

Rename `PiPLSComponentResult` to `PiPLSSelection` and
`PiPLSPredictorRankProfile.selected_result` to `selection`. No compatibility aliases remain at
version `0.0.0`.

The final vocabulary is:

```python
selection = search.select(...)
selection = model.selection_
selection = report.selection
selection = rank_profile.selection
```

The class may remain implemented in `component_path.py`; this decision does not require a module
move.

### Remove duplicated global best attributes

Remove these fitted `PiPLSSearchCV` attributes:

```text
best_index_
best_score_
best_n_components_
best_predictor_rank_
best_params_
```

The canonical global configured-score optimum is:

```python
best = search.select(rule="best_score")
```

Private index calculation may remain where implementation needs it. `rank_test_score` remains in
`cv_results_` because it describes every candidate rather than one duplicated selected row.

### Simplify candidate parameter columns

Keep the stable, pipeline-independent candidate columns:

```text
n_components
predictor_rank
```

Remove these duplicated `cv_results_` entries:

```text
params
param_<n_components key>
param_<predictor_rank key>
```

Retain candidate scores, split-level values, response-standardized MSE fields, ranks, and timing
statistics. `cv_results_` remains the advanced detailed surface.

### Use `X` and `Y` as the dataset matrix names

Retain:

```python
data.X
data.Y
```

Remove:

```python
data.data
data.target
```

Retain `n_samples`, `n_features`, and `n_targets`, which are useful dataset-level dimensions rather
than aliases for another semantic object.

### Remove unused shape-only inspection properties

Remove properties that merely repeat public array shapes and have no maintained application use:

```text
BiplotCoordinates.n_samples
BiplotCoordinates.n_features
LatentStructure.n_samples
LatentStructure.n_features
LatentStructure.n_targets
ObservationDiagnostics.n_samples
PiPLSDisplayFactors.n_features
PiPLSDisplayFactors.n_targets
PredictionDiagnostics.n_samples
PredictionDiagnostics.n_targets
```

Retain `LatentStructure.n_components` and `PiPLSDisplayFactors.n_components`, which are used to
control component-oriented analysis. Retain `BiplotCoordinates.scaling_factors`, because it records
the actual numerical balancing transformation rather than a repeated shape.

All five numerical inspection functions remain public:

```text
biplot_coordinates()
latent_structure()
observation_diagnostics()
pipls_display_factors()
prediction_diagnostics()
```

### Reduce top-level exports

The final top-level estimator surface is:

```python
from pipls import (
    PiPLSRegression,
    PiPLSSearchCV,
    PredictorRankSupportWarning,
)
```

Returned result classes remain public from their focused modules:

```python
from pipls.component_path import (
    PiPLSComponentPath,
    PiPLSSelection,
    PiPLSPredictorRankProfile,
)
from pipls.decomposition import PiPLSDecomposition
from pipls.validation import PiPLSOOFReport
```

This removes convenience re-exports, not the result types themselves.

### Retain direct-construction validation and caller-owned plotting

Immutable public records continue to validate direct construction and pickle reconstruction. This
is part of their safety boundary even when ordinary users receive them from package functions.

The package continues to expose no public plotting module. Maintained examples keep caller-owned
Matplotlib rendering, and no active numerical inspection or private maintained plotting function is
removed merely because it is specialized.

### Implementation sequence

Implement this cleanup in seven reviewable patches:

1. establish this decision and the guide-layer target;
2. simplify `PiPLSOOFReport` and remove impossible absent-array handling;
3. rename `PiPLSComponentResult` to `PiPLSSelection` and rank-profile `selected_result` to
   `selection`;
4. remove public fitted-search `best_*` attributes while preserving best-score selection numerics;
5. remove duplicated `cv_results_` parameter representations;
6. remove dataset aliases and unused shape-only inspection properties;
7. reduce top-level exports, synchronize all documentation, complete active-surface audits, and
   mark this decision implemented.

Each patch must leave the package, examples, and documentation internally consistent. No aliases,
deprecation warnings, or compatibility shims are introduced.

## Implementation status

Patches 1–3 are complete. The decision and guide-layer target are recorded; `PiPLSOOFReport`
requires OOF prediction and count arrays while exposing selection metrics only through
`report.selection`; and the complete active API now uses `PiPLSSelection` and rank-profile
`selection` terminology without changing search numerics. Patch 4 should next remove the public
fitted-search `best_*` attributes while preserving best-score selection behavior.

This decision refines Decisions 0051, 0052, 0071, 0072, 0086, 0087, 0090, 0094, 0111, 0116, 0129,
0134, 0138, 0140, and 0143. It preserves their numerical, immutability, direct-construction,
selection, OOF, dataset-resource, and caller-owned-rendering boundaries while narrowing duplicated
access paths.

## Consequences

- Ordinary modeling code gains fewer competing names and shorter result contracts.
- Selection-only workflows retain `search.select()` without presenting it as a mandatory refit
  step.
- OOF report consumers access selection evidence through `report.selection` and can rely on OOF
  arrays being present.
- Search users obtain the global configured-score optimum through the same selection object used by
  all other rules.
- Advanced candidate analysis retains detailed scores, split values, timings, and stable direct
  parameter columns.
- Reference-dataset examples use `X` and `Y` consistently across mathematical prose and code.
- Numerical inspection and caller-owned plotting capabilities remain intact.
- Pre-release duplication is removed rather than deprecated or aliased.
