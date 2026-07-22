# Decision 0067: direct Sugarcane reference workflow

## Status

Accepted and implemented.

## Context

The Sugarcane example already exposed its data reading and model choice, but the analytical path
between fitting and plotting remained indirect. It converted `component_path_` into a pandas table,
wrote that table to CSV, reread it through a plotting helper, generated seven additional
post-analysis tables, reread those tables, reconstructed immutable inspection results, and only then
rendered a multipage report.

Those round trips obscured the ordinary programming workflow. The public package already provides
immutable component-path, factor, latent-structure, and prediction-diagnostic results that can be
plotted directly. Sugarcane is the smallest spectral example and is therefore the appropriate first
reference implementation for the Phase F4 simplification.

## Decision

`examples/11_sugarcane_real_data.py` owns one visible linear workflow:

1. read `X.csv` and `Y.csv` directly with pandas;
2. fit `PiPLSPathCV(refit=False)`;
3. plot `component_path_` directly with ordinary Matplotlib;
4. retrieve the chosen scalar result with `for_n_components()`;
5. fit one fixed `PiPLSRegression` model;
6. calculate five-fold non-shuffled predictions with scikit-learn `cross_val_predict()`;
7. calculate `PiPLSDisplayFactors`, `LatentStructure`, and `PredictionDiagnostics` in memory;
8. create every figure explicitly and call public one-axis plotters through `ax=`.

The example writes only these final PDF figures:

```text
component_path.pdf
pipls_factors.pdf
prediction_diagnostics.pdf
latent_structure.pdf
coefficients.pdf
```

It writes no generated analytical CSV files, does not reconstruct inspection results from tables,
and does not import the component-path, fixed-model OOF, or post-analysis artifact helpers. The
wavelength coordinate continues to come directly from the `X.csv` column headers. Prediction
figures retain the label `selection-conditioned OOF predictions` because the fixed pair is chosen
from a path evaluated on the same observations.

## Consequences

- The complete Sugarcane workflow is readable without following support modules.
- `component_path_` and the immutable inspection objects are the analytical interface; pandas is
  used only for the committed input tables.
- Figure composition, legends, titles, saving, and closing remain example-owned, while package
  plotters remain atomic.
- Pulp and Tobacco retain their existing CSV and report helpers until their scheduled Phase F4
  migrations. This decision does not expand or preserve those helpers as a long-term public pattern.
- No package API or numerical behavior changes.
