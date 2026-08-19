# Public API contract

## Namespace boundary

The top-level package exports only:

```python
from pipls import PiPLSRegression, PiPLSSearchCV, PredictorRankSupportWarning
```

`__version__` is package metadata. Public result records and utilities are imported from focused
modules:

```python
from pipls.component_path import (
    PiPLSComponentPath,
    PiPLSPredictorRankEvidence,
    PiPLSPredictorRankProfile,
    PiPLSSelection,
)
from pipls.decomposition import PiPLSDecomposition
from pipls.validation import PiPLSOOFReport
```

`pipls.datasets`, `pipls.inspection`, and `pipls.metrics` expose exactly their declared `__all__`
names. Private modules, including `_core`, `_cv_engine`, `_model_selection`, `_result_validation`,
and `_sklearn_compat`, are not compatibility surfaces.

Mathematical prose uses `Y`; estimator signatures follow scikit-learn's `fit(X, y)` convention,
where `y` may be one- or two-dimensional.

## `PiPLSRegression`

The fixed estimator fits one explicit pair. Required keyword-only parameters are
`n_components` and `predictor_rank`; neither has a default, `None`, or automatic sentinel. Optional
parameters are `scale`, `scale_x`, `scale_y`, `copy`, `svd_solver`, and `random_state`.

Validation rules:

- ranks are positive Python or NumPy integers; booleans and floats are invalid;
- `n_components <= predictor_rank`;
- `predictor_rank <= min(n_features, n_samples - 1)` after centering and no greater than the
  verified numerical rank;
- `scale` and `copy` are booleans;
- `scale_x` and `scale_y` are booleans or `None`; `None` inherits `scale` for the corresponding
  block;
- `svd_solver` is `"full"`, `"randomized"`, or `"auto"`;
- `random_state` accepts `None`, a valid integer seed, or NumPy `RandomState`.

A direct fit emits `PredictorRankSupportWarning` when fewer than three observations support each
retained predictor-rank direction. The warning is diagnostic and does not alter the requested rank.
Controlled search probes, candidate fits, OOF fits, and full-data refits suppress only this expected
warning category.

The estimator provides PLS-style `fit`, `predict`, `transform`, `fit_transform`,
`inverse_transform`, and scalar R2 `score`. It supports feature names and inherited scikit-learn
`set_output()` behavior. Standard fitted scores, loadings, rotations, coefficients, intercept, and
means/scales remain available.

Every fit centers both blocks. `scale` remains the backward-compatible default for both predictor
and response scaling. Non-`None` `scale_x` and `scale_y` values override that policy independently,
allowing a pipeline transformer to own predictor scaling while Pi-PLS retains response scaling.

`decomposition_` is an immutable `PiPLSDecomposition` containing:

- `predictor_directions` (`P`);
- `dilation` (the diagonal entries of `D`);
- `response_directions` (`Q`);
- predictor numerical-rank evidence and tolerance;
- resolved predictor SVD solver;
- derived centered/scaled regression map.

Private construction matrices and redundant aliases are not exposed. A directly fitted fixed model
has no selection provenance.

## `PiPLSSearchCV`

The search owns cross-validated path evaluation. It accepts a direct `PiPLSRegression`, a
scikit-learn pipeline with one terminal Pi-PLS step, or `None` for the default template. Arbitrary
nested meta-estimators are unsupported.

Important defaults and controls:

- `n_components_values="all"` scans every component count compatible with the active predictor-rank policy;
- `predictor_rank_values=None` optimizes predictor rank over the complete hard-feasible integer domain;
- `search_method="exhaustive"` is the default and evaluates every admissible predictor rank;
- `search_method="adaptive"` is an explicit deterministic approximation over the same rank domain;
- `predictor_rank_values="epv"` fixes one rank from the EPV-inspired rule;
- a one-element sequence fixes one rank and a longer sequence defines an explicit optimization domain;
- `max_predictor_rank=None` imposes no user cap; a positive integer adds an explicit upper restriction;
- `samples_per_predictor_rank=10.0` is used only by EPV; a nondefault value is invalid outside EPV;
- `cv=5` remains the ordinary validation default;
- `scoring="neg_response_standardized_mse"` resolves to the package scorer;
- standard scorer names, callables, and `None` remain accepted.

Decision 0148 defines these constructor controls:

```python
predictor_rank_relative_tolerance=None
predictor_rank_absolute_tolerance=np.inf
```

They are separate from `select()` and `refit()` tolerances. `None` resolves to the square root of
float64 machine epsilon; positive-infinity absolute tolerance disables that cap. Nondefault values
are invalid for fixed and EPV predictor-rank policies because those policies perform no conditional
predictor-rank optimization. Fixed and EPV policies are compatible with the constructor-default
`search_method="exhaustive"` and report exhaustive coverage because each compatible component count
has only one rank candidate.

`fit()` materializes one validation split set, evaluates candidate clones, and stores immutable
candidate/path evidence. It does not retain the training matrices or fit a final model.

### Selection

```python
selection = search.select(rule="best_score")
selection = search.select(
    rule="minimum_cv_mse",
    relative_tolerance=None,
    absolute_tolerance=np.inf,
)
selection = search.select(n_components=4)
```

Exactly one of `rule` and `n_components` is required. Selection performs no fitting, rescoring,
split materialization, or mutation.

`best_score` returns the maximum configured-score row on the predictor-rank-conditioned component
path. `minimum_cv_mse` returns the smallest component count satisfying both CV-MSE tolerance caps.
`relative_tolerance=None` resolves to the square root of float64 machine epsilon; positive-infinity
absolute tolerance disables that cap. Tolerance arguments are invalid for other rules and manual
selection.

Both named rules operate on the predictor-rank-conditioned component path. An unretained global
candidate from `cv_results_` is not eligible for a named component-count rule. Manual lookup also
returns the conditioned row.

### Full-data refit

```python
selection = search.select(rule="minimum_cv_mse")
model = search.refit(X, y, selection=selection)

model = search.refit(X, y, rule="minimum_cv_mse")
model = search.refit(X, y, n_components=4)
```

Exactly one of `selection`, `rule`, and `n_components` determines the stored row. A supplied
selection must be a `PiPLSSelection` exactly compatible with the fitted search, and nondefault
component-count tolerances are invalid because the row is already resolved. `refit()` uses the same
compatibility definition as `oof_report()`, clones the configured estimator or pipeline, fits it on
the supplied full data, and attaches the exact supplied object or resolved immutable row to the
returned outer estimator as `selection_`. It does not mutate the fitted search, retain the supplied
data, or store the returned model. To fit an exact manually specified
`(n_components, predictor_rank)` pair, use `PiPLSRegression` directly.

### OOF report

```python
selection = search.select(rule="minimum_cv_mse")
report = search.oof_report(X, y, selection=selection)
model = search.refit(X, y, selection=selection)
```

The selection must match the same fitted search exactly, including tolerance provenance. The caller
must provide the same row-aligned observations because the search stores indices and shapes, not the
original values. The method reuses all materialized splits, averages repeated predictions per row,
and returns `PiPLSOOFReport`. It neither selects again nor fits a full-data model.

### Fitted search evidence

The main public evidence is:

- `component_path_`: one conditionally selected predictor-rank row per evaluated component count;
- `predictor_rank_profile(n_components)`: all evaluated predictor ranks at one component count;
- `cv_results_`: stable direct parameter columns, configured scores, split values, CV-MSE fields,
  ranks, and timings;
- `max_predictor_rank_`, `n_splits_`, `n_targets_`, `search_is_exhaustive_`, and standard
  scikit-learn feature metadata.

There is no fitted `best_*` state and no automatic final estimator.

## Path and selection records

`PiPLSComponentPath` contains aligned read-only arrays for component counts, conditionally selected
predictor ranks, configured mean scores, mean CV-MSE, split SD, plus path-wide predictor-rank policy
and split count.

`PiPLSPredictorRankProfile` contains the evaluated predictor ranks and aligned evidence for one
component count. `reference_selection` is the exact configured-score choice; `selection` is the
smallest evaluated tolerance-qualified rank and carries `predictor_rank_evidence`.

Immutable `PiPLSPredictorRankEvidence` contains the exact-reference rank and score,
reference CV-MSE mean and SD, resolved predictor-rank tolerances, and derived `score_threshold`.
Optimized path rows and selections carry this evidence; fixed and EPV policies carry `None`.

`PiPLSSelection` contains one evaluated pair, predictor-rank policy, configured score, CV-MSE mean,
CV-MSE split SD, split count, and optional rule provenance. A `minimum_cv_mse` result additionally
contains:

- `reference_minimum`: exact unruled minimum row;
- resolved `relative_tolerance`;
- resolved `absolute_tolerance`;
- derived `cv_mse_threshold`.

There is no standard-error property or selection rule.

## OOF result

`PiPLSOOFReport` contains:

- the exact compatible `selection`;
- ordered `oof_predictions`;
- per-observation `oof_prediction_counts`;
- optional pooled OOF R2.

Selection metrics are accessed through `report.selection`; they are not duplicated on the report.
Rows without validation coverage have zero counts and NaN predictions.

OOF reports carry no validation-protocol classifier. Generic predictions, counts, coverage, and
pooled OOF R2 remain public and protocol-neutral. `cv` continues to accept compatible splitters and
explicit split iterables, but the package does not identify a protocol from its materialized split
structure.

## Metrics

`pipls.metrics` exports `response_standardized_mse` and
`neg_response_standardized_mse`. They use response scales learned from the fitted Pi-PLS estimator
or terminal Pi-PLS pipeline step and average uniformly across observations and response columns.

## Datasets and synthetic generators

`pipls.datasets` exports:

```python
PiPLSDataset
PiPLSLatentGeometryTruth
PiPLSRegressionTruth
load_pulp
load_sugarcane
load_tobacco
make_pipls_latent_geometry
make_pipls_regression
make_pipls_train_test
```

`PiPLSDataset` uses canonical matrix attributes `X` and `Y`; metadata, names, sample identifiers,
provenance, and optional truth records are immutable. Reference loaders return a fresh immutable
object by default or fresh read-only `(X, Y)` arrays with `return_X_y=True`. They perform no hidden
preprocessing and require no network access or pandas.

The synthetic generators are deterministic for fixed validated seeds and return immutable truth
records where applicable. They are package utilities, not benchmark or publication-result APIs.

## Inspection

`pipls.inspection` exports immutable result types and five pure functions:

```python
pipls_display_factors
latent_structure
biplot_coordinates
observation_diagnostics
prediction_diagnostics
```

`pipls_display_factors()` is Pi-PLS-specific and returns sign-oriented copies of `P`, dilation, `Q`,
and derived `QD` while preserving the regression map.

`latent_structure()`, `biplot_coordinates()`, and `observation_diagnostics()` accept compatible
fitted PLS-family models through public fitted operations. `prediction_diagnostics()` accepts
observed and predicted responses explicitly and requires a `PredictionKind` provenance label.
Residuals are `observed - predicted`; the immutable result also derives response-wise standardized
RMSE and response-wise $R^2$ for those supplied predictions.

Inspection results are defensive, read-only, finite, directly validated, and pickle-safe. The
package exposes no plotting module, Matplotlib artist result, or public `plot_*` helper.

## Example and rendering boundary

Examples 04--06 fit Pi-PLS paths, create one immutable selection, optionally compute a matching
OOF report, refit that same selection, inspect the fitted model, and render final PDFs directly with
Matplotlib. Pulp uses 50 repeated five-fold splits and averages ten OOF predictions per
observation. Sugarcane and Tobacco use seeded shuffled five-fold CV. Tobacco demonstrates a
constructor-level `predictor_rank_relative_tolerance=0.10` and a separate component-count
`relative_tolerance=0.10`; its figures and console output identify both exact references,
thresholds, and retained choices.

The example-local ordinary-PLS path helper is not package API. Optional Matplotlib and `adjustText`
dependencies remain outside the runtime dependency set.

## Explicit public exclusions

The package does not expose:

- automatic outer validation;
- a final estimator from `search.fit()`;
- general weighted fitting or metadata routing;
- generic dataset download/registry APIs;
- a package-owned block-aware scaling class or block-definition API;
- public plotting or report-composition helpers;
- adaptive-search execution history as result state;
- private scorer/preprocessing state needed only internally;
- aliases for retired pre-release names.
