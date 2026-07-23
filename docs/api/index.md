# API reference

The generated reference documents supported public objects directly from their Python docstrings.
Core estimators are available from `pipls`; focused numerical inspection, dataset, and metric
tools live in dedicated submodules. Rendering is caller-owned Matplotlib code.

## Start with the estimators

- [Fixed regression](regression.md)
- [Path selection](path.md)
- [Troubleshooting](../troubleshooting.md)

## Which result object should I use?

| Object | Obtained from | Main purpose |
|---|---|---|
| [`PiPLSComponentPath`](path.md#pipls.PiPLSComponentPath) | `search.component_path_` | Compare component counts and their conditionally selected predictor ranks |
| [`PiPLSComponentResult`](path.md#pipls.PiPLSComponentResult) | `path.for_n_components(h)` | Retrieve one evaluated fixed rank pair |
| [`PiPLSPredictorRankProfile`](path.md#pipls.PiPLSPredictorRankProfile) | `search.predictor_rank_profile(h)` | Inspect all predictor ranks evaluated at one component count |
| [`PiPLSValidationReport`](validation.md#pipls.PiPLSValidationReport) | `search.validation_report_` | Inspect validation provenance, coverage, and selected-candidate diagnostics |
| [`PiPLSDecomposition`](decomposition.md#pipls.PiPLSDecomposition) | `model.decomposition_` | Access interpretable predictor rotations, dilation, response rotations, and rank diagnostics |
| [`LatentStructure`](inspection.md#pipls.inspection.LatentStructure) | `latent_structure(model)` | Access scores, loadings, rotations, and coefficients for PLS-family inspection |
| [`PiPLSDisplayFactors`](inspection.md#pipls.inspection.PiPLSDisplayFactors) | `pipls_display_factors(model.decomposition_)` | Obtain display-oriented $P$, $D$, $Q$, and $QD$ factors |
| [`BiplotCoordinates`](inspection.md#pipls.inspection.BiplotCoordinates) | `biplot_coordinates(model)` | Construct balanced two-component score-loading coordinates |
| [`ObservationDiagnostics`](inspection.md#pipls.inspection.ObservationDiagnostics) | `observation_diagnostics(model)` | Inspect score distance and X-reconstruction residuals |
| [`PredictionDiagnostics`](inspection.md#pipls.inspection.PredictionDiagnostics) | `prediction_diagnostics(Y, Y_pred, ...)` | Inspect predictions, residuals, and response-standardized errors |
| [`PiPLSDataset`](datasets.md#pipls.datasets.PiPLSDataset) | dataset construction or generator output | Carry validated arrays, labels, provenance, and metadata |
| [`PiPLSSyntheticTruth`](datasets.md#pipls.datasets.PiPLSSyntheticTruth) | `synthetic.truth` | Inspect the known latent structure of generated data |

## Rendering boundary

The API ends at immutable numerical results. Pi-PLS provides no plotting submodule or public
`plot_*` functions. Maintained examples pass result arrays to ordinary Matplotlib calls.
`biplot_coordinates()` is retained because coordinate balancing is numerical; optional
`adjustText` placement operates on the resulting Matplotlib text artists.

## Other reference pages

- [Decomposition](decomposition.md)
- [Validation report](validation.md)
- [Warnings](exceptions.md)
- [Inspection](inspection.md)
- [Dataset containers and synthetic data](datasets.md)
- [Metrics](metrics.md)
