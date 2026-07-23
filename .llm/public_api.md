# Public API contract

## Current top-level API

```python
from pipls import (
    PiPLSDecomposition,
    PiPLSPathCV,
    PiPLSRegression,
    PiPLSValidationReport,
    StatisticalSupportWarning,
)
```

The generated reference under `docs/api/` documents exactly these supported top-level objects and
the declared public names from `pipls.inspection`, `pipls.datasets`, and `pipls.metrics`;
`__version__` remains package metadata rather than an API reference page.
Private modules and `pipls.model_selection` are not reference surfaces.

`PiPLSRegression` fits one explicit fixed pair `(n_components, predictor_rank)` and performs no
cross-validation or parameter selection. `PiPLSPathCV` is the standard package workflow for the
bounded triangular scan and conditional predictor-rank selection. Public scoring callables remain
available from `pipls.metrics`.

## Public names

| Public name | Mathematical notation |
|---|---|
| `n_components` | $h$ |
| `predictor_rank` | $r_\pi$ |
| `samples_per_predictor_rank` | $c$ in `PiPLSPathCV` |
| `predictor_rank_` | fitted explicit $r_\pi$ |
| `max_predictor_rank_` | centered algebraic limit on the fixed estimator; search ceiling on the path |

Do not expose constructor aliases named `h`, `r_pi`, or `c`.

## Fixed-estimator validation and warning contract

`PiPLSRegression` constructor parameters are `n_components`, `scale`, `copy`, `predictor_rank`,
`svd_solver`, and `random_state`.

- `n_components` and `predictor_rank` are positive Python or NumPy integers; booleans and
  integral-valued floats are invalid.
- `n_components <= predictor_rank`.
- `predictor_rank <= min(n_features, n_samples - 1)` after centering and must not exceed the
  verified numerical rank.
- `random_state` accepts an integer in $[0, 2^{32}-1]$, a NumPy `RandomState`, or `None`;
  the default integer `0` is reproducible and `None` uses NumPy global state.
- `scale` and `copy` are Python or NumPy booleans.

A direct fixed fit emits `StatisticalSupportWarning` when $n/r_\pi<3$. This warning is diagnostic;
it does not choose or cap the rank. `PiPLSPathCV` suppresses only this expected warning inside its
controlled feature probes, candidate fits, optional OOF fits, and selected full-data refit. Other
warning categories remain visible.

Both public `fit()` methods are transactional. A failed initial fit or refit removes all fitted
attributes, including any state from an earlier successful fit. A successful fixed fit exposes only
finite fitted arrays. `copy=False` may reuse independent writable arrays, but read-only arrays and
overlapping predictor/response storage are copied as needed to preserve correctness.

## Model-internal standardization contract

Every `PiPLSRegression.fit` estimates `x_mean_` and `y_mean_` from the observations supplied to
that fit. `scale=True` estimates safe sample-standard-deviation vectors with `ddof=1` and
standardizes both blocks; `scale=False` still centers and stores unit scales. During path
selection, every candidate clone learns these statistics only from its training fold. The optional
final refit learns them again from all observations supplied to `PiPLSPathCV.fit`.

## Predictor SVD policy

`svd_solver` accepts `"full"`, `"randomized"`, or `"auto"`; the default is `"auto"`.
`random_state=0` makes randomized decomposition reproducible; a NumPy `RandomState` and `None` are
also accepted. Only the first SVD of the centered/scaled predictor matrix may be randomized. The
response-subspace and coupling SVDs remain exact.

## Fitted fixed-estimator behavior

The estimator provides PLS-style `fit`, `predict(X, copy=True)`,
`transform(X, y=None, copy=True)`, tuple-valued `fit_transform(X, y)`, `inverse_transform`, and
scalar R2 `score`. It supports feature names and pandas output. Standard PLS-style fitted
attributes, coefficients, and scores remain available. The frozen `decomposition_` object exposes
only interpretable predictor rotations, dilation values, response rotations, numerical-rank
diagnostics, the resolved predictor solver, and the derived centered/scaled regression map. The
construction matrices $\Pi$, $C$, and $W$, the redundant diagonal matrix $D$, and symbolic aliases
remain private.

The fixed estimator does not expose `cv_results_`, `best_params_`, OOF predictions, validation
reports, predictor-rank search diagnostics, or scorer plumbing. The response scale required by the
default scorer is private fitted state. `predictor_rank_` is the requested fitted integer and
`max_predictor_rank_` is `min(n_features, n_samples - 1)` for the supplied centered training
data.

## Path-analysis API

`PiPLSPathCV` owns all package model selection. It defaults to
`n_components_values="all"`, which resolves every admissible component count. Explicit integer
sequences request a subset; `None` is not a component-path alias. It also defaults to adaptive
`search_method="auto"`; explicit `"optimal"` exhaustively evaluates every admissible pair. The
surface satisfies

\[
1 \le h \le \min(q,r_{\pi,\max}), \qquad h \le r_\pi \le r_{\pi,\max}.
\]

The default ceiling uses total supplied $n$ for the support term with
`samples_per_predictor_rank=5`, while centered fold dimensions remain hard feasibility caps. The
class accepts a direct fixed `PiPLSRegression` or a pipeline ending in one, materializes one CV
split set, clones fixed candidates, and optionally refits the selected pair.

The default `scoring` value is the public callable
`pipls.metrics.neg_response_standardized_mean_squared_error`. Ordinary scikit-learn scorer names,
other callables, and `None` remain accepted. Conditional and overall selections among evaluated
candidates maximize the configured mean test score. Under the default scorer this is
equivalent to minimizing mean response-standardized MSE among evaluated candidates; adaptive
search makes no claim about
unevaluated admissible pairs.

Public path attributes include standard candidate-level search results in `cv_results_`, global
`best_*` selection attributes, `path_search_exhaustive_`, optional refitted estimators, immutable
`validation_report_`, and the canonical immutable `component_path_` result. OOF arrays and their
coverage counts live only in `validation_report_`. Validated input grids, adaptive-search batch
history, candidate counters, direct-rank parameter aliases, and matrix-shaped score/MSE aliases are
not public fitted state; advanced users can inspect aligned `cv_results_` columns when needed.
`PiPLSComponentPath` stores aligned read-only `n_components`,
`predictor_rank`, `predictor_rank_policy`, `mean_test_score`, `cv_mse_mean`, `cv_mse_fold_sd`, and
`n_splits` arrays. `for_n_components()` returns a frozen `PiPLSComponentResult` with the aligned
scalar values. The numeric predictor rank is present for every component count.

`PiPLSPathCV.predictor_rank_profile(h)` derives an immutable
`PiPLSPredictorRankProfile` on demand from `cv_results_`. Its aligned read-only arrays contain only
predictor ranks actually evaluated at `h`, sorted in ascending order, and its `selected` field is
the same scalar result returned by `component_path_.for_n_components(h)`. The profile does not add
another fitted attribute or stored search representation. Selection maximizes the configured mean
test score; only the default scorer makes this equivalent to minimizing mean response-standardized
CV-MSE.
Refit-dependent delegated methods are absent when `refit=False`. Output-container configuration is
owned by the estimator template and preserved through cloning and refit; the path object does not
add a separate `set_output` layer.

Group-aware splitters and keyword-only `groups` belong to `PiPLSPathCV.fit`, not to the fixed
estimator. `return_oof_predictions` and selection-conditioned reporting likewise belong only to the
path interface.

## E1 dataset and synthetic-data API

Dataset functionality is public from the dedicated `pipls.datasets` namespace and is declared by
that module's `__all__`:

```python
from pipls.datasets import (
    PiPLSDataset,
    PiPLSSyntheticTruth,
    make_pipls_regression,
    make_pipls_train_test,
)
```

`PiPLSDataset` is an optional immutable in-memory container, primarily useful for package-owned
synthetic data and structured experiments. Plain arrays and data frames passed directly to
`fit(X, Y)` remain the primary real-data interface. The container stores read-only `float64` `X` and
2D `Y`, unique feature/target/sample names, required provenance, recursively frozen metadata, and
optional synthetic truth. `data` and `target` are scikit-learn-style aliases. Required provenance
keys are `source`, `license`, `citation`, and `version`.

`make_pipls_regression` creates one side-effect-free dataset with local seeded random generation.
It supports shared, predictor-specific, and response-specific latent ranks; scalar or per-direction
strengths; normal or uniform source distributions; scalar or per-variable observed scales; and
scalar or separate predictor/response noise. `random_state=0` is the deterministic default and
must be an unsigned 32-bit integer. Each sample block must contain more rows than the larger
centered latent rank requested for `X` or `Y`.

`make_pipls_train_test` creates two datasets from one shared loading/strength/scale model and
independent train/test score and noise draws. It performs no fitted preprocessing and the training
block does not depend on the requested test size.

`PiPLSSyntheticTruth` exposes read-only latent scores, loading blocks, signal/noise matrices,
strengths, and scales. Loading blocks that are structurally absent are explicit zeros.

No metadata file, registry lookup, or package-owned loader is required for real-data fitting.
Users read and prepare `X` and `Y` with ordinary domain-appropriate code. Repository examples
must show these steps directly rather than hiding them behind convenience utilities.

## Accepted model-inspection boundary

Decision 0045 distinguishes method-specific Pi-PLS factorization inspection from shared PLS-family
analysis. Final public names for $P$, $D$, and $Q$ inspection retain an explicit `pipls` marker.
Scores, loadings, coefficients, biplots, observation diagnostics, and prediction diagnostics use
estimator-neutral names and may accept compatible fitted `PLSRegression` or `PiPLSRegression`
objects. Numbered examples apply these shared tools only to the selected Pi-PLS model. Ordinary PLS
remains available in the dedicated component-path comparison example and declared comparator benchmarks. Examples 10–12 evaluate only Pi-PLS paths.

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
`PiPLSDecomposition`, copies $P$, $D$, and $Q$, chooses deterministic display signs from the first
largest-magnitude predictor entry, and preserves $PDQ^\mathsf{T}$. It returns $P$, the dilation
vector, $Q$, $QD$, and the applied signs as read-only arrays.

`prediction_diagnostics()` accepts one- or two-dimensional observed and predicted responses,
normalizes outputs to two dimensions, uses residuals $y-\hat y$, and applies observed-response
sample centers and standard deviations with `ddof=1`. It returns original and standardized arrays,
response centers and scales, response-wise standardized RMSE, and one of the explicit prediction
provenance labels defined by `PredictionKind`. Constant response columns and ambiguous labels are
rejected.

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

The package exposes no `pipls.plotting` module and no public `plot_*` convenience functions.
The immutable numerical results are the compatibility surface; Matplotlib artists, styles, and
`adjustText` label positions are caller-owned and are not package results.
`PiPLSDisplayFactors`, `LatentStructure`, `ObservationDiagnostics`, `PredictionDiagnostics`, and
`BiplotCoordinates` expose the numerical quantities required for rendering. Maintained examples
create Matplotlib figures and axes directly, including component and response selection, physical
coordinates, grouped-bar widths, labels, legends, titles, saving, and closing.

Annotated biplots are rendered from `BiplotCoordinates` with optional `adjustText` label placement.
Matplotlib and `adjustText` remain optional under the examples, docs, and development extras and are
not imported by the runtime package. No estimator method, fitted attribute, path-search parameter,
or top-level export is added by the rendering layer.

## Example workflow boundary

Example 09 owns the explicit Pi-PLS-versus-ordinary-PLS path comparisons and plots both immutable
component paths directly in memory. Pulp, Sugarcane, and Tobacco use `PiPLSPathCV(refit=False)`, plot
`component_path_` directly, read the selected pair through `for_n_components()`, fit one fixed
`PiPLSRegression`, and calculate five-fold non-shuffled predictions through scikit-learn
`cross_val_predict()`. They render immutable Pi-PLS factors, latent structure, observation diagnostics, and prediction
diagnostics directly with Matplotlib and write only final PDF figures. Pulp also reads the
conditional predictor-rank profile from `cv_results_` for the chosen component count. Tobacco uses
full predictor SVD, direct observation diagnostics, and caller-owned source-order response
pagination through multipage PDFs.

The package exposes no dataset I/O, tutorial workflow, predictor-rank-profile object, or
component-path plotting helper. The comparison-only `PLSComponentPath` remains example-local.
