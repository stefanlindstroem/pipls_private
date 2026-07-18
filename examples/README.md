# Examples

These concise executable examples demonstrate ordinary package use. They are not manuscript-figure
or publication-result workflows.

- `07_advanced_cv.py`: grouped and advanced cross-validation workflows.
- `08_synthetic_data.py`: deterministic train/test generation with shared latent structure.
- `10_pulp_real_data.py`: direct pandas reading, separate Pi-PLS and standard PLS path CSVs, a
  CSV-derived comparison PDF, and a fixed Pulp Pi-PLS model chosen from the recorded path.
- `11_sugarcane_real_data.py`: the same comparison workflow for high-dimensional LabSpec sugarcane
  spectra and four responses.
- `12_tobacco_real_data.py`: adaptive predictor-rank scanning with explicit full predictor SVD, a
  standard PLS comparison path, and a separately chosen fixed Pi-PLS model.
- `pls_component_path.py`: evaluate the scikit-learn PLSRegression NIPALS path with fold-local
  standardization and write its canonical CSV.
- `plot_component_path.py`: read the Pi-PLS and optional PLS CSV paths and generate their PDF view.

Install the data and plotting dependencies before running the real-data path examples:

```bash
python -m pip install -e ".[examples]"
```

The Pulp, Sugarcane, and Tobacco examples deliberately separate two stages:

1. `PiPLSPathCV(refit=False)` writes `examples/results/<dataset>_component_path.csv`, while
   scikit-learn `PLSRegression` writes `<dataset>_pls_component_path.csv` for the same folds and
   component counts. The PDF reads both CSV files and labels the paths `$\Pi$-PLS` and
   `PLS (NIPALS)`.
2. A visible `CHOSEN_N_COMPONENTS` constant selects one Pi-PLS row, and a separate
   `PiPLSRegression` fixes both recorded Pi-PLS ranks for the final full-data fit.

The CSV files are canonical. Error bars in the PDF show fold-to-fold SD, not a confidence
interval. Predictor-rank annotations apply only to Pi-PLS. Generated files under `examples/results/` are ignored by Git.

Real-data examples must show the ordinary I/O used to create `X` and `Y` in the example itself. Do
not route example data through a package registry, generic loader, or hidden data-reading helper.
