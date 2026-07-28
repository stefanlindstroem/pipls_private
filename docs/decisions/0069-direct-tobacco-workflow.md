# Decision 0069: direct Tobacco workflow

## Status

Accepted and implemented.

## Context

The Tobacco example was the final numbered real-data workflow that converted
in-memory numerical results to long-form pandas tables, wrote those tables as
CSV files, reread them, reconstructed immutable inspection objects, and only
then created figures. It also depended on a private fixed-model OOF helper even
though scikit-learn already provides the required prediction operation.

Tobacco still requires dataset-specific application logic that should remain
visible: a full predictor SVD, adaptive predictor-rank scanning, a decreasing
wavenumber coordinate, thirteen responses divided into deterministic
source-order pages, and raw observation diagnostics.

## Decision

`examples/12_tobacco_real_data.py` owns one visible linear workflow:

1. read `X.csv` and `Y.csv` directly with pandas;
2. form the physical wavenumber coordinate from the `X.csv` headers;
3. fit `PiPLSSearchCV(refit=False)` with a full-SVD estimator template;
4. plot `component_path_` directly with Matplotlib;
5. choose $h=8$ with `for_n_components()`;
6. fit one fixed full-SVD `PiPLSRegression` model;
7. calculate five-fold OOF predictions with `cross_val_predict()`;
8. calculate display factors, latent structure, prediction diagnostics, and
   observation diagnostics in memory;
9. partition the thirteen responses in source order, five per page;
10. create every figure, axis, pagination loop, PDF write, and close operation
    explicitly in the numbered example.

The example writes only final PDF files:

```text
component_path.pdf
pipls_factors.pdf
prediction_diagnostics.pdf
latent_structure.pdf
coefficients.pdf
```

`prediction_diagnostics.pdf` and `coefficients.pdf` each contain three response
pages. The first two pages contain five responses and the final page contains
the remaining three responses. The spectral predictor coordinate remains in
its source decreasing-wavenumber order.

Delete the obsolete example support modules:

```text
examples/_support/fixed_model_oof.py
examples/_support/post_analysis_artifacts.py
```

The comparison-specific `pls_component_path.py` and `plot_component_path.py`
remain until Patch 20e because example 09 still consumes their CSV contracts.

## Consequences

- All complete Pi-PLS real-data examples now use in-memory public results.
- No numbered fixed-model analysis converts immutable results to CSV and back.
- Tobacco keeps its full-SVD, response-pagination, spectral-axis, and raw
  observation-diagnostic behavior without a repository-specific report layer.
- The two multipage PDFs make response pagination explicit while keeping the
  final output inventory compact.
- The default test suite protects the direct workflow structurally rather than
  executing the expensive Tobacco analysis.
- No package numerical behavior or public estimator contract changes.
