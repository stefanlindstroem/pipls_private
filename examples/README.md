# Examples

These concise executable examples demonstrate ordinary package use. They are not manuscript-figure
or publication-result workflows.

- `07_advanced_cv.py`: grouped and advanced cross-validation workflows.
- `08_synthetic_data.py`: deterministic train/test generation with shared latent structure.
- `09_model_inspection.py`: a fixed Pi-PLS fit, $P$/$D$/$QD$ interpretation, and external-test
  prediction diagnostics on deterministic synthetic data.
- `10_pulp_real_data.py`: direct pandas reading, separate Pi-PLS and standard PLS path CSVs, a
  CSV-derived comparison PDF, and a fixed Pulp Pi-PLS model chosen from the recorded path.
- `11_sugarcane_real_data.py`: the same comparison workflow for high-dimensional LabSpec sugarcane
  spectra and four responses.
- `12_tobacco_real_data.py`: adaptive predictor-rank scanning with explicit full predictor SVD, a
  standard PLS comparison path, and a separately chosen fixed Pi-PLS model.
- `pls_component_path.py`: reusable standard-PLS path evaluation for the real-data examples.
- `plot_component_path.py`: reusable CSV-to-PDF plotting for the Pi-PLS and PLS paths.

Install the data and plotting dependencies before running the real-data examples:

```bash
python -m pip install -e ".[examples]"
make examples
```

`make examples` runs all numbered examples in order. It is intentionally separate from
`make check`: the real-data analyses can be slow and generate application artifacts under
`examples/results/`.

The current Pulp, Sugarcane, and Tobacco examples deliberately separate two stages:

1. `PiPLSPathCV(refit=False)` uses the default `n_components_values="all"` and produces one
   Pi-PLS row per admissible component count, while scikit-learn
   `PLSRegression` produces a comparison path for the same folds and component counts. The example
   writes both DataFrames as canonical CSV files and calls the plotting function on those files.
2. A visible `CHOSEN_N_COMPONENTS` constant selects one Pi-PLS CSV row, and a separate
   `PiPLSRegression` fixes both recorded ranks for the final full-data fit.

The examples import the two small helper functions directly. They do not launch subprocesses or
hide data reading behind a package loader. The CSV files remain canonical; the PDF is only a view of
them. Error bars show fold-to-fold SD, not a confidence interval. Predictor-rank annotations apply
only to Pi-PLS. Generated files under `examples/results/` are ignored by Git.

Decision 0042 adds a third, separately implemented stage for post-fit analysis. The existing
`pls_component_path.py` and `plot_component_path.py` remain selection-diagnostic helpers. Reusable
Pi-PLS computations belong in `pipls.inspection`, and `pipls.plotting` now renders decomposition
and prediction-diagnostic figures from those immutable results. Dataset-specific fixed-model OOF
loops, pandas tables, CSV writing, physical axes, pagination, and multipage reports remain
example-local.

Full-data decomposition, score, loading, and coefficient plots will be labeled as fitted-model
interpretation. Prediction and residual plots will accept explicit predictions and record whether
they are fitted, fixed-parameter OOF, selection-conditioned OOF, or external-test values. The
real-data examples will use `selection-conditioned OOF predictions` after component counts have
been chosen from paths computed on the same observations.

Real-data examples must show the ordinary I/O used to create `X` and `Y` in the example itself. Do
not route example data through a package registry, generic loader, or hidden data-reading helper.
