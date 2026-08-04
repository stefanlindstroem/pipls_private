# API reference

The generated reference documents supported public objects directly from their Python docstrings.
Core estimators and the public warning are available from `pipls`; their defining modules expose
only the corresponding public object. Returned result records, numerical inspection, dataset, and
metric tools live in focused submodules. Rendering is caller-owned.

## Mathematical notation and Python names

Mathematical sections use $\mathbf{X}$ for the predictor matrix and $\mathbf{Y}$ for the
response matrix. Python
call signatures follow the scikit-learn convention `fit(X, y)`: `y` may be either a one-dimensional
response or a two-dimensional multivariate response matrix. Names such as `y_pred`, `y_scores_`,
and `y_loadings_` follow the same programming convention and do not imply a scalar response.

For Pi-PLS, public `n_components` counts paired latent modes $h$, and `predictor_rank` is the
retained predictor-subspace dimension $r_\pi$. The factor arrays $\mathbf{P}$ and
$\mathbf{Q}$ are orthonormal
predictor and response directions; they are distinct from reconstruction loadings.

## Start with the estimators

- [Fixed regression](regression.md)
- [Path selection](path.md)
- [Troubleshooting](../troubleshooting.md)

## Which result object should I use?

| Object | Obtained from | Main purpose |
|---|---|---|
| [`PiPLSComponentPath`](path.md#pipls.component_path.PiPLSComponentPath) | `search.component_path_` | Compare paired-mode counts through aligned numerical evidence |
| [`PiPLSSelection`](path.md#pipls.component_path.PiPLSSelection) | `search.select(...)`, `model.selection_`, or a post-fit report | Retrieve one evaluated fixed rank pair and its diagnostics |
| [`PiPLSPredictorRankProfile`](path.md#pipls.component_path.PiPLSPredictorRankProfile) | `search.predictor_rank_profile(h)` | Inspect all predictor ranks evaluated at one paired-mode count |
| [`PiPLSOOFReport`](path.md#pipls.validation.PiPLSOOFReport) | `search.oof_report(X, Y, selection=...)` | Inspect ordered OOF predictions and coverage for one existing selection |
| [`PiPLSDecomposition`](regression.md#pipls.decomposition.PiPLSDecomposition) | `model.decomposition_` | Access predictor directions, dilation, response directions, and rank diagnostics |
| [`LatentStructure`](inspection.md#pipls.inspection.LatentStructure) | `latent_structure(model)` | Access scores, loadings, and coefficients for PLS-family inspection |
| [`PiPLSDisplayFactors`](inspection.md#pipls.inspection.PiPLSDisplayFactors) | `pipls_display_factors(model.decomposition_)` | Obtain display-oriented $\mathbf{P}$, $\mathbf{D}$, $\mathbf{Q}$, and $\mathbf{Q}\mathbf{D}$ factors |
| [`BiplotCoordinates`](inspection.md#pipls.inspection.BiplotCoordinates) | `biplot_coordinates(model)` | Construct balanced two-component score-loading coordinates |
| [`ObservationDiagnostics`](inspection.md#pipls.inspection.ObservationDiagnostics) | `observation_diagnostics(model)` | Inspect score distance and X-reconstruction residuals |
| [`PredictionDiagnostics`](inspection.md#pipls.inspection.PredictionDiagnostics) | `prediction_diagnostics(Y, Y_pred, ...)` | Inspect predictions, residuals, and response-standardized errors |
| [`PiPLSDataset`](datasets.md#pipls.datasets.PiPLSDataset) | a named loader, dataset construction, or generator output | Carry validated arrays, labels, provenance, and metadata |
| [`PiPLSRegressionTruth`](datasets.md#pipls.datasets.PiPLSRegressionTruth) | `synthetic.truth` | Inspect the known latent structure of generated data |
| [`PiPLSLatentGeometryTruth`](datasets.md#pipls.datasets.PiPLSLatentGeometryTruth) | `make_pipls_latent_geometry(...).truth` | Inspect the manuscript-oriented Gaussian latent geometry |

`PiPLSSearchCV.select()` returns complete stored component rows by component count or by the
`best_score`, `minimum_cv_mse`, and temporary `one_standard_error` rules without fitting or mutating
the search. The minimum-CV-MSE rule accepts simultaneous relative and absolute tolerances and
retains their complete selection provenance. Rule scope and scorer qualification are described under
[search-owned selection rules](../path_analysis.md#search-owned-selection-rules).

Pulp, Sugarcane, and Tobacco are available as named package-owned datasets through
[`load_pulp()`](datasets.md#pipls.datasets.load_pulp),
[`load_sugarcane()`](datasets.md#pipls.datasets.load_sugarcane), and
[`load_tobacco()`](datasets.md#pipls.datasets.load_tobacco). No generic dataset registry is part of
the runtime API.

## Rendering boundary

The API ends at immutable numerical results. Pi-PLS provides no plotting submodule. Maintained
examples pass result arrays to ordinary Matplotlib calls. `biplot_coordinates()` is retained because
coordinate balancing is numerical; optional `adjustText` placement operates on the resulting text
artists.
