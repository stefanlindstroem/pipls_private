# Public API contract

## Current top-level API

```python
from pipls import (
    PiPLSComponentPath,
    PiPLSComponentResult,
    PiPLSDecomposition,
    PiPLSSearchCV,
    PiPLSPredictorRankProfile,
    PiPLSRegression,
    PiPLSValidationReport,
    PredictorRankSupportWarning,
)
```

The generated reference under `docs/api/` documents exactly these supported top-level objects and
the declared public names from `pipls.inspection`, `pipls.datasets`, and `pipls.metrics`;
`__version__` remains package metadata rather than an API reference page.
Private modules and `pipls.model_selection` are not reference surfaces.

Mathematical documentation denotes the response matrix by $Y$. Public estimator methods follow the
scikit-learn `fit(X, y)` naming convention, so `y` may be a one-dimensional response or a
two-dimensional multivariate response matrix. Related public names such as `y_pred`, `y_scores_`,
and `y_loadings_` retain the same convention. User-facing plot labels should describe responses
and residuals without implying that a multivariate response is scalar.

`PiPLSRegression` fits one explicit fixed pair `(n_components, predictor_rank)` and performs no
cross-validation or parameter selection. Both rank parameters are required keyword-only
constructor arguments; neither has a default or accepts a missing-value sentinel. `PiPLSSearchCV` is
the standard package workflow for the bounded triangular scan and conditional predictor-rank selection. Public scoring callables remain
available from `pipls.metrics`.

## Public names

| Public name | Mathematical meaning |
|---|---|
| `n_components` | number of paired latent modes $h$ |
| `predictor_rank` | retained predictor-subspace dimension $r_\pi$ |
| `samples_per_predictor_rank` | $c$ in `PiPLSSearchCV` |
| `predictor_rank_` | fitted explicit $r_\pi$ |
| `max_predictor_rank_` | centered algebraic limit on the fixed estimator; search ceiling on the path |

Do not expose constructor aliases named `h`, `r_pi`, or `c`.

The immutable decomposition fields are `predictor_directions` and `response_directions`. The
estimator also exposes the standard PLS-style fitted attributes `x_rotations_` and `y_rotations_`,
which reference the same arrays. Mathematical prose calls their $P$ and $Q$ columns predictor and
response directions. These directions are distinct from the reconstruction loadings exposed as
`x_loadings_` and `y_loadings_`.

## Fixed-estimator validation and warning contract

`PiPLSRegression` constructor parameters are required keyword-only `n_components` and
`predictor_rank`, followed by optional `scale`, `copy`, `svd_solver`, and `random_state`.

- `n_components` and `predictor_rank` have no defaults and accept no `None` or automatic sentinel.
- They are positive Python or NumPy integers; booleans and
  integral-valued floats are invalid.
- `n_components <= predictor_rank`.
- `predictor_rank <= min(n_features, n_samples - 1)` after centering and must not exceed the
  verified numerical rank.
- `random_state` accepts an integer in $[0, 2^{32}-1]$, a NumPy `RandomState`, or `None`;
  the default integer `0` is reproducible and `None` uses NumPy global state.
- `scale` and `copy` are Python or NumPy booleans.

A direct fixed fit emits `PredictorRankSupportWarning` when $n/r_\pi<3$. This warning is diagnostic;
it does not choose or cap the rank. `PiPLSSearchCV` suppresses only this expected warning inside its
controlled feature probes, candidate fits, explicit OOF reports, and post-fit full-data refits.
Other warning categories remain visible.

Both public `fit()` methods are transactional. A failed search fit or fixed-estimator fit removes
fitted attributes from the object being fitted, including any state from an earlier successful fit.
A failed `PiPLSSearchCV.refit()` leaves the fitted search unchanged because it operates on a fresh
clone. A successful fixed fit exposes only finite fitted arrays. `copy=False` may reuse independent
writable arrays, but read-only arrays and overlapping predictor/response storage are copied as needed
to preserve correctness.

## Model-internal standardization contract

Every `PiPLSRegression.fit` estimates `x_mean_` and `y_mean_` from the observations supplied to
that fit. `scale=True` estimates safe sample-standard-deviation vectors with `ddof=1` and
standardizes both blocks; `scale=False` still centers and stores unit scales. During path
selection, every candidate clone learns these statistics only from its training fold. An explicit
post-fit `PiPLSSearchCV.refit(X, y, ...)` learns them again from all observations supplied to that
operation.

## Predictor SVD policy

`svd_solver` accepts `"full"`, `"randomized"`, or `"auto"`; the default is `"auto"`.
`random_state=0` makes randomized decomposition reproducible; a NumPy `RandomState` and `None` are
also accepted. Only the first SVD of the centered/scaled predictor matrix may be randomized. The
response-subspace and coupling SVDs remain exact.

## Fitted fixed-estimator behavior

The estimator provides PLS-style `fit`, `predict(X, copy=True)`,
`transform(X, y=None, copy=True)`, tuple-valued `fit_transform(X, y)`, `inverse_transform`, and
scalar R2 `score`. It supports feature names and inherited scikit-learn `set_output()`
configuration for transform containers, including pandas output. Standard PLS-style fitted
attributes, coefficients, and scores remain available. The frozen `decomposition_` object exposes
only interpretable predictor directions, dilation values, response directions, numerical-rank
diagnostics, the resolved predictor solver, and the derived centered/scaled regression map. The
construction matrices $\Pi$, $C$, and $W$, the redundant diagonal matrix $D$, and symbolic aliases
remain private.

The fixed estimator does not expose `cv_results_`, `best_params_`, OOF predictions, validation
reports, predictor-rank search diagnostics, or scorer plumbing. The response scale required by the
default scorer is private fitted state. `predictor_rank_` is the requested fitted integer and
`max_predictor_rank_` is `min(n_features, n_samples - 1)` for the supplied centered training
data.

## Path-analysis API

`PiPLSSearchCV` owns all package model selection. It defaults to
`n_components_values="all"`, which resolves every admissible component count. Explicit integer
sequences request a subset; `None` is not a component-path alias. It also defaults to adaptive
`search_method="auto"`; explicit `"optimal"` exhaustively evaluates every admissible pair. The
surface satisfies

\[
1 \le h \le \min(q,r_{\pi,\mathrm{max}}), \qquad h \le r_\pi \le r_{\pi,\mathrm{max}}.
\]

The default ceiling uses total supplied $n$ for the support term with
`samples_per_predictor_rank=5`. Centered fold dimensions and the minimum predictor rank verified
after fold-local pipeline preprocessing and terminal-estimator preprocessing are hard feasibility
caps. Explicit component and predictor-rank sequences are validated against the resolved ceiling
before scoring. The class accepts a direct fixed `PiPLSRegression` or a pipeline ending in one,
materializes one CV split set, and clones fixed candidates for path evaluation.

Final full-data fitting is an explicit post-fit operation:

```python
search = PiPLSSearchCV(cv=cv).fit(X, y)
model = search.refit(X, y, rule="one_standard_error")
model = search.refit(X, y, n_components=4)
```

`refit()` requires exactly one of `rule` and `n_components`. The supported named rules are
`"best_score"`, `"minimum_cv_mse"`, and `"one_standard_error"`. Manual component selection uses the
predictor rank already selected conditionally for that component-path row. The method clones the
configured direct estimator or terminal-Pi-PLS pipeline, fits that clone on the supplied full data,
and returns it. It does not mutate the search, retain the supplied data, or attach the returned model
to search state. Exact manual `(n_components, predictor_rank)` pairs are fitted directly with
`PiPLSRegression`.

The default `scoring` value is the stable package string
`"neg_response_standardized_mse"`, which resolves to the public callable
`pipls.metrics.neg_response_standardized_mse`. Ordinary scikit-learn scorer names, other callables,
and `None` remain accepted. Conditional and overall selections among evaluated candidates maximize
the configured mean test score. Under the default scorer this is equivalent to minimizing mean
response-standardized MSE among evaluated candidates; adaptive search makes no claim about
unevaluated admissible pairs.

Explicit selected-row inspection is a post-fit operation:

```python
selected = search.select(rule="one_standard_error")
selected = search.select(n_components=4)
```

The method requires exactly one selection input and returns one immutable stored
`PiPLSComponentResult`. It performs no fitting, rescoring, split materialization, or mutation. The
private `SelectionRule` vocabulary and search-owned resolver are shared with `refit()` and
`validation_report()`.

Explicit selection-conditioned OOF reporting is a post-fit operation:

```python
report = search.validation_report(X, y, rule="one_standard_error")
report = search.validation_report(X, y, n_components=4)
```

The method requires exactly one selection input, uses the same stored-row resolver as `refit()`, and
reuses defensive read-only copies of the exact validation indices materialized by `fit()`. It always
returns ordered OOF predictions and prediction counts, averages repeated predictions, marks uncovered
rows with NaN and count zero, computes pooled OOF $R^2$ only over covered rows, and performs neither
candidate rescoring nor a full-data fit. It validates the fitted sample, feature, and response-column
shape but cannot compare values; callers must preserve original row alignment. The search stores the
split indices but not supplied training matrices or returned reports.

Public path attributes include standard candidate-level search results in `cv_results_`, global
`best_*` selection attributes, `search_is_exhaustive_`, and the canonical immutable
`component_path_` result. The search stores no selected row, validation report, or fitted final
model. `PiPLSValidationReport` composes one immutable `PiPLSComponentResult`; its `n_components`,
`predictor_rank`, `n_splits`, `mean_test_score`, and `cv_mse_mean` properties forward to that result
rather than duplicating state. `is_selection_conditioned` and `has_complete_oof_coverage` expose
provenance and coverage predicates.

Validated input grids, adaptive-search batch history, candidate counters, direct-rank parameter
aliases, matrix-shaped score/MSE aliases, returned fitted estimators, and supplied training matrices
are not public fitted state. Advanced users can inspect aligned `cv_results_` columns when needed.

`PiPLSComponentPath` stores aligned read-only `n_components`, `predictor_rank`,
`mean_test_score`, `cv_mse_mean`, and `cv_mse_fold_sd` arrays. The predictor-rank policy and number of
validation splits are path-wide Python scalars. It derives the aligned read-only
`cv_mse_standard_error` array from the stored population fold SD and shared split count. New code
retrieves one complete stored row through `search.select(...)`. The path-level
`for_n_components()`, `minimum_cv_mse_result()`, and `one_standard_error_result()` methods remain
temporarily for migration parity and preserve their exact stored-value, tie, standard-error, and
error contracts until Patch 4 removes them.

`best_index_`, `best_score_`, `best_params_`, `best_n_components_`, and
`best_predictor_rank_` always describe the global configured-score optimum. They are search evidence,
not a stored final estimator. The search exposes no `predict`, `transform`, `fit_transform`,
`inverse_transform`, `score`, or `get_feature_names_out` delegation. Call those methods on the
estimator or pipeline returned by `refit()`. Output-container configuration is owned by the estimator
template and is preserved through cloning; the path object adds no separate `set_output` layer.

`PiPLSSearchCV.predictor_rank_profile(h)` derives an immutable
`PiPLSPredictorRankProfile` on demand from `cv_results_`. Its aligned read-only arrays contain only
predictor ranks actually evaluated at `h`, sorted in ascending order. Its path-wide policy and split
count are scalars, and its `selected_result` property derives the same conditionally selected scalar
values as `search.select(n_components=h)` from immutable candidate state. The profile does not
add another fitted attribute or stored selected-row representation. It exposes an aligned read-only
`cv_mse_standard_error` property derived by the same contract as the component path. Selection
maximizes the configured mean test score; only the default scorer makes this equivalent to minimizing
mean response-standardized CV-MSE.

Decision 0140 Patch 2 is implemented. `search.select(rule=... or n_components=...)` is the new
search-owned selected-row lookup, the shared private rule vocabulary is `SelectionRule`, and
selection inspection, refitting, validation reporting, and predictor-rank profile composition use
search-owned helpers. The path-level methods remain temporarily until maintained consumers migrate
in Patch 3 and are removed in Patch 4.

All five top-level result records (`PiPLSDecomposition`, `PiPLSComponentResult`,
`PiPLSPredictorRankProfile`, `PiPLSComponentPath`, and `PiPLSValidationReport`) validate direct
construction, normalize accepted NumPy scalars to Python scalars, defensively copy arrays, and
reconstruct through the same validation path when unpickled. Invalid dimensions, nonfinite scores,
negative MSE summaries, unsupported policy values, and inconsistent OOF coverage are rejected.
Generated documentation keeps these records returned-first by suppressing constructor signatures.

Group-aware splitters and keyword-only `groups` belong to `PiPLSSearchCV.fit`, not to the fixed
estimator. Explicit selection-conditioned reporting belongs to `PiPLSSearchCV.validation_report()`;
no report is constructed or attached during search fitting.

## E1 dataset and synthetic-data API

Dataset functionality is public from the dedicated `pipls.datasets` namespace and is declared by
that module's `__all__`:

```python
from pipls.datasets import (
    PiPLSDataset,
    PiPLSLatentGeometryTruth,
    PiPLSRegressionTruth,
    load_pulp,
    make_pipls_latent_geometry,
    make_pipls_regression,
    make_pipls_train_test,
)
```

`load_pulp(*, return_X_y=False)` is implemented. It returns `PiPLSDataset` by default and a
pair of fresh read-only `float64` arrays when `return_X_y=True`. It is backed by installed package
resources, verifies resource and canonical-array integrity, performs no network access or
preprocessing, and is exported only from `pipls.datasets`; no `as_frame`, registry, or generic
loader is authorized. All maintained Pulp consumers use this loader, and the package resources are
the sole active Pulp matrix representation.

`PiPLSDataset` is an optional immutable in-memory container, primarily useful for package-owned
synthetic data and structured experiments. Plain arrays and data frames passed directly to
`fit(X, Y)` remain the primary real-data interface. The container stores read-only `float64` `X` and
2D `Y`, unique feature/target/sample names, required provenance, recursively frozen metadata, and
optional synthetic truth. Metadata arrays preserve non-object dtypes, are copied, and are made
read-only; object-dtype arrays are rejected because their Python elements cannot be frozen by
making the array container read-only. `data` and `target` are scikit-learn-style aliases. Required
provenance keys are `source`, `license`, `citation`, and `version`.

`make_pipls_regression` creates one side-effect-free dataset with local seeded random generation.
It supports shared, predictor-specific, and response-specific latent ranks; scalar or per-direction
strengths; normal or uniform source distributions; scalar or per-variable observed scales; and
scalar or separate predictor/response noise. `random_state=0` is the deterministic default and
must be an unsigned 32-bit integer. Each sample block must contain more rows than the larger
centered latent rank requested for `X` or `Y`.

`make_pipls_train_test` creates two datasets from one shared loading/strength/scale model and
independent train/test score and noise draws. It performs no fitted preprocessing and the training
block does not depend on the requested test size.

`make_pipls_latent_geometry` is a separate manuscript-aligned generator. It draws independent
standard-normal predictor-specific, shared, and response-specific score matrices; independent
standard-normal loading matrices; and independent Gaussian predictor/response noise. It applies no
score centering or standardization, loading orthonormalization, strength scaling, or observed-scale
transformation. Its loading matrices use manuscript orientation with latent dimensions on rows.

`PiPLSRegressionTruth` exposes read-only latent scores, contributing loading blocks, signal/noise
matrices, strengths, and scales for the configurable package generators. Structurally absent
cross-side loading blocks are not stored as redundant zero arrays. `PiPLSLatentGeometryTruth`
exposes the manuscript matrices directly and validates the two signal equations. Both truth forms
may be carried by `PiPLSDataset.truth`.

The served companion-manuscript synthetic-data guide must distinguish reproducing the exact
data-generating distribution, reproducing one seeded realization, and reproducing complete
publication results. It may document the known oracle dimensions
$r_\pi=d_{\mathrm{p}}+d_{\mathrm{s}}$ and $h=d_{\mathrm{s}}$ for
the synthetic experiments, but it must not redefine package search defaults or practical real-data
selection. Complete grids, comparator pipelines, and paper figure/table orchestration remain
downstream publication assets.

No metadata file, registry lookup, or package-owned loader is required for real-data fitting.
Users read and prepare `X` and `Y` with ordinary domain-appropriate code. Repository examples
must show these steps directly rather than hiding them behind convenience utilities.

## Accepted model-inspection boundary

Decision 0045 distinguishes method-specific Pi-PLS factorization inspection from shared PLS-family
analysis. Final public names for $P$, $D$, and $Q$ inspection retain an explicit `pipls` marker.
Scores, loadings, coefficients, biplots, observation diagnostics, and prediction diagnostics use
estimator-neutral names and may accept compatible fitted `PLSRegression` or `PiPLSRegression`
objects. Numbered examples apply these shared tools only to the selected Pi-PLS model. Ordinary PLS
remains available in the dedicated component-path comparison example. Examples 05–07 evaluate only Pi-PLS paths.

Decision 0042 introduced a staged inspection and plotting design. Decisions 0079--0082 establish
the final boundary: `pipls.inspection` owns pure NumPy computations and immutable result objects,
while examples and users render those arrays directly with Matplotlib.

`pipls.inspection` is implemented and exports from its own namespace:

```python
from pipls.inspection import (
    BiplotCoordinates,
    LatentStructure,
    ObservationDiagnostics,
    PiPLSDisplayFactors,
    PredictionDiagnostics,
    PredictionKind,
    pipls_display_factors,
    biplot_coordinates,
    latent_structure,
    observation_diagnostics,
    prediction_diagnostics,
)
```

These names are not top-level `pipls` exports. `pipls_display_factors()` accepts a
`PiPLSDecomposition`, copies $P$, $D$, and $Q$, and preserves $PDQ^\mathsf{T}$. Its default display
signs come from the first largest-magnitude predictor entry. A caller may instead supply a zero-based
`response_index` and request a positive or negative response orientation; exact zero anchor entries
fall back to the predictor convention. It returns $P$, the dilation vector, $Q$, and $QD$ as
read-only arrays; labels and sign bookkeeping remain outside the result record.

`prediction_diagnostics()` accepts one- or two-dimensional observed and predicted responses,
normalizes outputs to two dimensions, uses residuals $y-\hat y$, and applies observed-response
sample centers and standard deviations with `ddof=1`. `PredictionDiagnostics` accepts only the
independent observed values, predicted values, and prediction provenance; it derives and stores the
read-only residual, standardized arrays, response centers and scales, and response-wise standardized
RMSE. Constant response columns and ambiguous labels are rejected.

`latent_structure()` accepts a compatible fitted PLS-family model through the public
`x_scores_`, `x_loadings_`, `y_loadings_`, and `coef_` attributes. Both `PiPLSRegression` and
scikit-learn `PLSRegression` satisfy this contract. The returned arrays are defensive and read-only;
coefficients retain the common `(n_targets, n_features)` orientation.

`observation_diagnostics()` accepts a compatible fitted model with public X scores, X loadings,
`transform()`, and `inverse_transform()`. It calculates squared score distance relative to the
fitted training-score covariance and the row-wise squared X-reconstruction residual from the public
transform/inverse-transform round trip. The result contains read-only raw arrays and no theoretical
probability limits.

`biplot_coordinates()` accepts a `LatentStructure` and exactly two zero-based components. It returns balanced read-only sample and predictor coordinates that preserve the selected
$TP^\mathsf{T}$ reconstruction.

All five inspection records validate direct construction, store defensive read-only arrays, and
revalidate through pickle reconstruction. Display-factor weighting and prediction-diagnostic
quantities are derived from independent constructor state rather than accepted redundantly. Their
helper functions guarantee finite public arrays:
range-safe scaled calculations are used where ordinary means, norms, covariance products, squared
residuals, or RMSE calculations could overflow, and an unrepresentable derived quantity raises a
clear `ValueError` instead of returning `inf` or `nan`.

The package exposes no `pipls.plotting` module and no public `plot_*` convenience functions.
The immutable numerical results are the compatibility surface; Matplotlib artists, styles, and
`adjustText` label positions are caller-owned and are not package results.
`PiPLSDisplayFactors`, `LatentStructure`, `ObservationDiagnostics`, `PredictionDiagnostics`, and
`BiplotCoordinates` expose the numerical quantities required for rendering.
`PiPLSDisplayFactors.weighted_response_directions` is derived as a checked read-only $QD$ array
from the stored response directions and dilation rather than accepted as independent constructor
state. Maintained examples
create Matplotlib figures and axes directly, including component and response selection, physical
coordinates, grouped-bar widths, labels, legends, titles, saving, and closing.

Annotated biplots are rendered from `BiplotCoordinates` with optional `adjustText` label placement.
Matplotlib and `adjustText` remain optional under the `examples`, `docs`, and `dev` extras and are
not imported by the runtime package. No `data` extra is exposed; real-data reading is user-owned.
No estimator method, fitted attribute, path-search parameter, or top-level export is added by the
rendering layer.

## Example workflow boundary

Example 04 owns the explicit Pi-PLS-versus-ordinary-PLS path comparisons and plots both immutable
component paths directly in memory. Pulp, Sugarcane, and Tobacco use the default path-evaluating
`PiPLSSearchCV()`, plot
`component_path_` directly, optionally read a scalar row for annotations, fit the chosen row through
`search.refit(...)`, and obtain five-fold seeded shuffled predictions through
`search.validation_report(...)`. The report reuses the exact explicit
`KFold(n_splits=5, shuffle=True, random_state=0)` partition materialized by the path search. They
render immutable Pi-PLS factors, latent structure, observation
diagnostics, and prediction diagnostics directly with Matplotlib and write only final PDF figures.
Pulp also reads the conditional predictor-rank profile through `predictor_rank_profile()` for the
chosen component count. Tobacco uses
full predictor SVD, direct observation diagnostics, and caller-owned source-order response
pagination through multipage PDFs.

The package exposes no dataset I/O, tutorial workflow, predictor-rank-profile object, or
component-path plotting helper. The comparison-only `PLSComponentPath` remains example-local.
