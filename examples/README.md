# Examples

These concise executable examples demonstrate ordinary package use. They are not manuscript-figure
or publication-result workflows.

- `07_advanced_cv.py`: grouped and advanced cross-validation workflows.
- `08_synthetic_data.py`: deterministic train/test generation with shared latent structure.
- `10_pulp_real_data.py`: direct pandas reading, separate Pi-PLS and standard PLS path CSVs, fixed
  full-data interpretation models, selection-conditioned OOF predictions, seven canonical
  post-analysis CSV files, and a multipage report reconstructed from them.
- `11_sugarcane_real_data.py`: direct pandas reading, Pi-PLS and standard PLS paths, fixed
  interpretation models, selection-conditioned OOF predictions, seven canonical post-analysis
  CSV files, and a wavelength-aware report reconstructed from them.
- `12_tobacco_real_data.py`: adaptive predictor-rank scanning with explicit full predictor SVD, a
  standard PLS comparison path, and a separately chosen fixed Pi-PLS model.
- `pls_component_path.py`: reusable standard-PLS path evaluation for the real-data examples.
- `plot_component_path.py`: reusable CSV-to-PDF plotting for the Pi-PLS and PLS paths.
- `fixed_model_oof.py`: example-local cloning and aligned OOF prediction for already fixed models.
- `post_analysis_artifacts.py`: canonical post-analysis table construction, CSV writing, rereading,
  and multipage report composition.

Install the data and plotting dependencies before running the real-data examples:

```bash
python -m pip install -e ".[examples]"
make examples
```

The Pulp and Sugarcane post-analysis examples show label acquisition as a separate I/O step:

```python
X = pd.read_csv(DATA_DIR / "X.csv")
Y = pd.read_csv(DATA_DIR / "Y.csv")
predictor_names = X.columns.astype(str).tolist()
response_names = Y.columns.astype(str).tolist()
```

These names are then passed explicitly to `pipls.inspection`, `pipls.plotting`, and the
example-owned artifact helpers. Users whose arrays do not carry column headers can obtain
equivalent lists from a schema or other domain metadata.

`make examples` runs all existing numbered examples in filename order; numbering gaps are
permitted when an obsolete example is removed. It is intentionally separate from
`make check`: the real-data analyses can be slow and generate application artifacts under
`examples/results/`.

The current Pulp, Sugarcane, and Tobacco examples begin with two common stages:

1. `PiPLSPathCV(refit=False)` uses the default `n_components_values="all"` and produces one
   Pi-PLS row per admissible component count, while scikit-learn
   `PLSRegression` produces a comparison path for the same folds and component counts. The example
   writes both DataFrames as canonical CSV files and calls the plotting function on those files.
2. A visible `CHOSEN_N_COMPONENTS` constant selects one Pi-PLS CSV row, and a separate
   `PiPLSRegression` fixes both recorded ranks for the final full-data fit.

The Pulp and Sugarcane examples add a third stage. Visible fixed Pi-PLS and PLS parameter choices
are cloned inside the same five non-shuffled folds, producing `selection-conditioned OOF
predictions`. Each example writes seven long-form CSV files under its dataset-specific
post-analysis directory, then rereads those files to generate `post_analysis.pdf`. Full-data
decomposition, score, loading, and coefficient tables remain interpretation artifacts; OOF rows
remain prediction diagnostics. Sugarcane reads the strictly increasing 780--2500 nm coordinate
from the `X.csv` headers and explicitly requests line rendering for predictor directions, X
loadings, and response-specific coefficients.

The display-standardized columns in `predictions.csv` use the complete observed-response matrix.
They are intended for a common response display and do not reproduce the fold-local standardization
used by the component-path loss.

The examples import the small helper functions directly. They do not launch subprocesses or
hide data reading behind a package loader. The CSV files remain canonical; the PDF is only a view of
them. Error bars show fold-to-fold SD, not a confidence interval. Predictor-rank annotations apply
only to Pi-PLS. Generated files under `examples/results/` are ignored by Git.

Decision 0042 adds a third, separately implemented stage for post-fit analysis. The existing
`pls_component_path.py` and `plot_component_path.py` remain selection-diagnostic helpers. Reusable
Pi-PLS and ordinary PLS computations belong in `pipls.inspection`. `pipls.plotting` now renders
Pi-PLS decomposition and prediction-diagnostic figures together with ordinary PLS score, X- and
Y-loading, and coefficient figures from immutable results. Dataset-specific fixed-model OOF loops,
pandas tables, CSV writing, physical axes, pagination, and multipage reports remain example-local.

Full-data decomposition, score, loading, and coefficient plots are fitted-model interpretation.
Prediction and residual plots accept explicit predictions and record whether they are fitted,
fixed-parameter OOF, selection-conditioned OOF, or external-test values. The Pulp and Sugarcane
real-data examples use `selection-conditioned OOF predictions` after component counts have been
chosen from paths computed on the same observations. Tobacco will adopt the same provenance
contract when its post-analysis stage is integrated.

Real-data examples must show the ordinary I/O used to create `X` and `Y` in the example itself. Do
not route example data through a package registry, generic loader, or hidden data-reading helper.
