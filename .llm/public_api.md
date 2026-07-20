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
the declared public names from `pipls.inspection`, `pipls.plotting`, `pipls.datasets`, and
`pipls.metrics`; `__version__` remains package metadata rather than an API reference page.
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
attributes, coefficients, and scores remain available. Pi-PLS factorization matrices, dilation
values, numerical-rank diagnostics, and the resolved predictor solver are canonicalized only in the
frozen `decomposition_` object; duplicate top-level symbolic and diagnostic aliases are not public.

The fixed estimator does not expose `cv_results_`, `best_params_`, OOF predictions, validation
reports, or predictor-rank search diagnostics. `response_scale_for_scoring_` remains available for
package scorers. `predictor_rank_` is the requested fitted integer and `max_predictor_rank_` is
`min(n_features, n_samples - 1)` for the supplied centered training data.

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

Public path attributes include standard search results, `best_pipls_`, conditional path and surface
diagnostics, immutable `validation_report_`, optional OOF outputs, and the canonical
`component_path_results_` table. The numeric predictor rank is present in every component-path row.
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

Decision 0042 accepts two public submodules for staged implementation:

- `pipls.inspection` for pure NumPy computations and immutable result objects;
- `pipls.plotting` for optional Matplotlib figures built from explicit computed results.

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

`pipls.plotting` is implemented and exports from its own namespace:

```python
from pipls.plotting import (
    PredictorStyle,
    plot_pipls_decomposition,
    plot_biplot,
    plot_coefficients,
    plot_observation_diagnostics,
    plot_scores,
    plot_x_loadings,
    plot_y_loadings,
    plot_prediction_diagnostics,
)
```

`plot_pipls_decomposition()` accepts `PiPLSDisplayFactors`, an explicit `"bar"` or `"line"`
predictor style, optional zero-based component indices, and caller-supplied scientific labels or a
physical predictor coordinate. It returns shared `predictor_directions`,
`weighted_response_directions`, and `dilation` axes. Selected components appear side by side within
named categorical bars or as overlaid lines on one physical predictor axis.
`plot_prediction_diagnostics()` accepts `PredictionDiagnostics`, optional zero-based response
indices, and required response labels. It returns named observed-versus-predicted, residual, and
RMSE axes and includes the stored prediction provenance in the title. `plot_scores()` renders
exactly two selected X-score columns. X and Y loadings place selected components together on one
axis; coefficient plots place selected responses together on one axis. Categorical displays require
caller-supplied variable names, while line displays require an explicit physical coordinate and axis
label. Components and responses are selected by explicit zero-based indices.
`plot_biplot()` renders balanced sample scores with named X-loading arrows and no response
arrows or confidence regions. `plot_observation_diagnostics()` renders one raw score-distance
versus X-reconstruction-residual
scatter plot without theoretical limits or automatic observation labels.

Matplotlib remains optional and is imported only when a plotting function is called. The functions
perform no file writing, call no display function, retain no models, and do not alter supplied
arrays. Component-path helpers remain example-local selection diagnostics, while dataset-specific
OOF loops, pandas tables, canonical CSV files, and PDF composition remain under `examples/`. No
estimator method, fitted attribute, path-search parameter, or top-level export is added by this
plotting layer.

## Example workflow boundary

Example 09 owns the explicit Pi-PLS-versus-ordinary-PLS path comparisons and their separate
canonical CSV files. Examples 10–12 use `PiPLSPathCV(refit=False)` to produce one Pi-PLS path, write
`component_path.csv` and `component_path.pdf` beside the post-analysis report, and fit the final
`PiPLSRegression` with the numeric predictor rank read from the chosen Pi-PLS row. The package
exposes no dataset I/O or plotting helper for this workflow.
