# Examples

The examples are arranged by user task rather than by implementation complexity. Start with the
literal-matrix fit, then move to selection, synthetic data, or the complete real-data workflows.
The complete workflows are intentionally more extensive than ordinary estimator use.

## Start here

- `01_minimal_fit_and_plot.py`: literal NumPy matrices, one fixed `PiPLSRegression` fit, predictions,
  and one Pi-PLS decomposition figure. It performs no cross-validation or parameter selection.

Run it with:

```bash
python -m pip install -e ".[examples]"
PYTHONPATH=src MPLBACKEND=Agg python examples/01_minimal_fit_and_plot.py
```

The script writes `examples/results/minimal_fit_and_plot.pdf`. Its predictor and response names are
ordinary Python lists, demonstrating that plotting labels may come from any explicit metadata
source rather than from pandas or CSV headers.

## Selection and synthetic-data examples

- `07_advanced_cv.py`: grouped and advanced cross-validation workflows.
- `08_synthetic_data.py`: deterministic train/test generation with shared latent structure.

## Complete reference workflows

- `10_pulp_real_data.py`: direct pandas reading, separate Pi-PLS and standard PLS path CSVs, fixed
  full-data interpretation models, selection-conditioned OOF predictions, seven canonical
  post-analysis CSV files, a balanced two-component score-loading biplot, and a multipage report.
- `11_sugarcane_real_data.py`: direct pandas reading, Pi-PLS and standard PLS paths, fixed
  interpretation models, selection-conditioned OOF predictions, seven canonical post-analysis
  CSV files, and a wavelength-aware report.
- `12_tobacco_real_data.py`: adaptive predictor-rank scanning with explicit full predictor SVD,
  fixed Pi-PLS and ordinary PLS interpretation models, selection-conditioned OOF predictions,
  decreasing-wavenumber spectral plots, deterministic response pagination, eight canonical
  post-analysis CSV files, and raw ordinary PLS observation diagnostics.

These are application analyses rather than introductory snippets. `make examples` runs every
numbered example in filename order, including the slower real-data workflows. It remains separate
from `make check`.

## Example support modules

The complete workflows import implementation support from `examples/_support/`:

- `pls_component_path.py`: ordinary-PLS path evaluation;
- `plot_component_path.py`: component-path CSV-to-PDF rendering;
- `fixed_model_oof.py`: cloning and aligned OOF prediction for already fixed models;
- `post_analysis_artifacts.py`: canonical table construction, CSV round trips, pagination, and
  multipage report composition.

The underscore-prefixed directory marks these files as support for the complete examples, not as
the shortest route to fitting Pi-PLS. They remain example-owned because they contain pandas I/O,
fixed-model OOF orchestration, physical-axis handling, and report composition. Reusable numerical
inspection belongs in `pipls.inspection`, and optional rendering belongs in `pipls.plotting`.

## Real-data workflow contract

The Pulp, Sugarcane, and Tobacco examples show label acquisition as a separate I/O step:

```python
X = pd.read_csv(DATA_DIR / "X.csv")
Y = pd.read_csv(DATA_DIR / "Y.csv")
predictor_names = X.columns.astype(str).tolist()
response_names = Y.columns.astype(str).tolist()
```

Users whose arrays do not carry column headers can supply equivalent lists from a schema, laboratory
information system, or other domain metadata. The package plotting API does not read files or
invent scientific variable names.

The three real-data examples share these stages:

1. `PiPLSPathCV(refit=False)` produces one Pi-PLS row per admissible component count, while
   scikit-learn `PLSRegression` produces a comparison path for the same folds and component counts.
   Both paths are written as canonical CSV files before plotting.
2. Visible component-count choices select fixed full-data Pi-PLS and PLS models for interpretation.
3. The same visible parameters are cloned inside five non-shuffled folds to produce
   `selection-conditioned OOF predictions`. Seven common long-form CSV files are written and
   reread before report generation.

Full-data decomposition, score, loading, coefficient, biplot, and observation-diagnostic figures
are interpretive. Prediction and residual figures retain explicit provenance. The display-standardized
columns in `predictions.csv` use the complete observed-response matrix and do not reproduce the
fold-local scaling used by the component-path loss.

Pulp reconstructs its biplot from `pls_scores.csv` and `pls_x_loadings.csv`. Sugarcane reads its
strictly increasing wavelength coordinate from `X.csv`. Tobacco preserves its decreasing
wavenumber coordinate, partitions all thirteen responses in source order, and adds
`pls_observation_diagnostics.csv`. Generated files under `examples/results/` are ignored by Git.
