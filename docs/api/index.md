# API reference

The generated reference documents supported public objects directly from their Python docstrings.
Core estimators are available from `pipls`; numerical inspection, dataset, and metric tools live in
focused submodules. Rendering is caller-owned.

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
| [`PiPLSComponentPath`](path.md#pipls.PiPLSComponentPath) | `search.component_path_` | Compare paired-mode counts and inspect minimum-CV-MSE or 1-SE result rows |
| [`PiPLSComponentResult`](path.md#pipls.PiPLSComponentResult) | explicit path lookup, a path recommendation method, or a post-fit report | Retrieve one evaluated fixed rank pair and its diagnostics |
| [`PiPLSPredictorRankProfile`](path.md#pipls.PiPLSPredictorRankProfile) | `search.predictor_rank_profile(h)` | Inspect all predictor ranks evaluated at one paired-mode count |
| [`PiPLSValidationReport`](path.md#pipls.PiPLSValidationReport) | `search.validation_report(X, Y, ...)` | Inspect validation provenance, coverage, and selected-candidate diagnostics |
| [`PiPLSDecomposition`](regression.md#pipls.PiPLSDecomposition) | `model.decomposition_` | Access predictor directions, dilation, response directions, and rank diagnostics |
| [`LatentStructure`](inspection.md#pipls.inspection.LatentStructure) | `latent_structure(model)` | Access scores, loadings, and coefficients for PLS-family inspection |
| [`PiPLSDisplayFactors`](inspection.md#pipls.inspection.PiPLSDisplayFactors) | `pipls_display_factors(model.decomposition_)` | Obtain display-oriented $\mathbf{P}$, $\mathbf{D}$, $\mathbf{Q}$, and $\mathbf{Q}\mathbf{D}$ factors |
| [`BiplotCoordinates`](inspection.md#pipls.inspection.BiplotCoordinates) | `biplot_coordinates(model)` | Construct balanced two-component score-loading coordinates |
| [`ObservationDiagnostics`](inspection.md#pipls.inspection.ObservationDiagnostics) | `observation_diagnostics(model)` | Inspect score distance and X-reconstruction residuals |
| [`PredictionDiagnostics`](inspection.md#pipls.inspection.PredictionDiagnostics) | `prediction_diagnostics(Y, Y_pred, ...)` | Inspect predictions, residuals, and response-standardized errors |
| [`PiPLSDataset`](datasets.md#pipls.datasets.PiPLSDataset) | dataset construction or generator output | Carry validated arrays, labels, provenance, and metadata |
| [`PiPLSRegressionTruth`](datasets.md#pipls.datasets.PiPLSRegressionTruth) | `synthetic.truth` | Inspect the known latent structure of generated data |
| [`PiPLSLatentGeometryTruth`](datasets.md#pipls.datasets.PiPLSLatentGeometryTruth) | `make_pipls_latent_geometry(...).truth` | Inspect the manuscript-oriented Gaussian latent geometry |

`PiPLSComponentPath.minimum_cv_mse_result()` and
`PiPLSComponentPath.one_standard_error_result()` return complete stored component rows for the two
reference rules without fitting or mutating the path search. Their scope and scorer qualification
are described under [result-object recommendations](../path_analysis.md#result-object-recommendations).

The Pulp, Sugarcane, and Tobacco [reference datasets](../datasets.md) are repository CSV assets,
not `PiPLSDataset` registry entries or package-owned loader results.

## Rendering boundary

The API ends at immutable numerical results. Pi-PLS provides no plotting submodule. Maintained
examples pass result arrays to ordinary Matplotlib calls. `biplot_coordinates()` is retained because
coordinate balancing is numerical; optional `adjustText` placement operates on the resulting text
artists.
