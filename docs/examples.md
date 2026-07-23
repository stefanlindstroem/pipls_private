# Examples

The numbered examples are executable workflows organized by purpose. Start with the
[synthetic tutorial](tutorials/synthetic.md) for the short selection-and-prediction sequence, then
continue with the [Pulp tutorial](tutorials/pulp.md) for a complete real-data analysis. This page is
a catalogue of the maintained scripts.

## Fixed fit

`examples/01_minimal_fit_and_plot.py` uses literal NumPy arrays, fits one fixed
`PiPLSRegression`, predicts responses, and creates a caller-owned $2\times2$ factor panel. It does
not perform parameter selection. The [fixed-regression reference](api/regression.md) documents
its estimator contract.

## Synthetic path selection

`examples/02_synthetic_path_selection.py` generates independent training and test blocks with known
shared, predictor-specific, and response-specific latent structure. It evaluates the component path,
inspects the conditional predictor-rank profile, fits one selected fixed model, reports held-out
$R^2$, and writes three final PDF figures. The [synthetic tutorial](tutorials/synthetic.md) extracts
its maintained code directly.

## Pi-PLS and ordinary PLS comparison

`examples/09_pls_path_comparison.py` evaluates matched component-count paths for Pulp, Sugarcane,
and Tobacco. It keeps both immutable paths in memory and writes one overlaid CV-MSE figure per dataset.
This comparison is optional and is not part of routine Pi-PLS fitting.

## Complete real-data analyses

The [dataset documentation](datasets.md) gives the original source, DOI, license, and repository
adaptation for each real-data integration. The three complete analyses evaluate one Pi-PLS
component path and fit one selected fixed model:

- `examples/10_pulp_real_data.py`: the direct tutorial workflow for named scalar predictors and
  responses, including the component path, an immutable conditional predictor-rank profile from
  `predictor_rank_profile()`, score-loading
  biplot, fixed-model inspection, and OOF diagnostics;
- `examples/11_sugarcane_real_data.py`: the direct reference workflow, with a visible in-memory
  component path, scikit-learn OOF prediction, wavelength-aware inspection, and five final PDF
  figures;
- `examples/12_tobacco_real_data.py`: decreasing-wavenumber plots, response pagination, and raw
  observation diagnostics.

Pulp writes six final PDF figures, including `predictor_rank_profile.pdf`; Sugarcane writes five;
Tobacco writes five, with three-page prediction-diagnostic and coefficient PDFs. No numbered
example writes a generated CSV file: committed `X.csv` and `Y.csv` tables are inputs, while every
figure is constructed directly from `component_path_`, scikit-learn OOF predictions, and
immutable inspection results. The example layer owns Matplotlib chart construction, physical
coordinates, subplot layouts, legends, figure-level titles, PDF output, and closing. Only the
Pi-PLS factor panels still use temporary one-axis convenience plotters.

The Pulp tutorial extracts its checked snippets directly from `examples/10_pulp_real_data.py`.
The sole module under `examples/_support/` evaluates the nontrivial fold-local ordinary-PLS path for
example 09. It is not required for ordinary estimator use.

## Run the examples

```bash
python -m pip install -e ".[examples]"
make examples
```

The complete real-data analyses are intentionally outside `make check` because they are application
workflows and may take substantially longer than the package test suite.
