# Examples

The numbered examples are executable workflows organized by purpose. Start with the
[Pulp tutorial](tutorials/pulp.md) when learning the complete model-development process; this page
is a catalogue of the maintained scripts.

## Fixed fit

`examples/01_minimal_fit_and_plot.py` uses literal NumPy arrays, fits one fixed
`PiPLSRegression`, predicts responses, and creates a caller-owned $2\times2$ factor panel. It does
not perform parameter selection. The same calculation appears in the [quickstart](quickstart.md).

## Synthetic train/test problem

`examples/08_synthetic_data.py` generates independent training and test blocks with known shared,
predictor-specific, and response-specific latent structure. It fits one fixed Pi-PLS model and
reports held-out $R^2$.

## Pi-PLS and ordinary PLS comparison

`examples/09_pls_path_comparison.py` evaluates matched component-count paths for Pulp, Sugarcane,
and Tobacco. It writes separate Pi-PLS and ordinary-PLS tables and one overlaid CV-MSE figure per
dataset. This comparison is optional and is not part of routine Pi-PLS fitting.

## Complete real-data analyses

The three complete analyses evaluate one Pi-PLS component path, fit one selected fixed model, and
write interpretation and prediction-diagnostic artifacts:

- `examples/10_pulp_real_data.py`: named scalar predictors and responses, including the maintained
  score-loading biplot;
- `examples/11_sugarcane_real_data.py`: wavelength-aware spectral directions, loadings, and
  coefficients;
- `examples/12_tobacco_real_data.py`: decreasing-wavenumber plots, response pagination, and raw
  observation diagnostics.

Each analysis writes `component_path.csv`, `component_path.pdf`, canonical post-analysis CSV tables,
and `post_analysis.pdf`. Selection and fitted-model interpretation remain separate artifacts. The
example layer owns subplot layouts, legends, figure-level titles, PDF output, and closing; package
plotters render one chart on one supplied axis.

The Pulp script and tutorial share the numerical workflow in
`examples/_support/pulp_workflow.py`. Other modules under `examples/_support/` provide example-only
OOF orchestration, path plotting, table writing, and report composition. They are not required for
ordinary estimator use.

## Run the examples

```bash
python -m pip install -e ".[examples]"
make examples
```

The complete real-data analyses are intentionally outside `make check` because they are application
workflows and may take substantially longer than the package test suite.
