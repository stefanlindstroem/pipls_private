# Examples

The numbered examples are organized by user task. Each one states what its data represent, what
question it addresses, and which outputs it produces.

## Minimal fixed fit

`examples/01_minimal_fit_and_plot.py` is the shortest complete route through the package. It uses
literal NumPy matrices, fits one fixed `PiPLSRegression`, predicts responses, and plots the fitted
$P D Q^{\mathsf T}$ decomposition. It performs no cross-validation or parameter selection.

The same workflow is reproduced in [`quickstart.md`](quickstart.md).

## Synthetic train/test problem

`examples/08_synthetic_data.py` generates independent training and test blocks with known latent
structure. Shared directions affect both predictors and responses, predictor-specific directions
add nuisance variation to $X$, and response-specific directions add variation to $Y$ that cannot be
predicted from $X$. The example fits one Pi-PLS model and reports held-out $R^2$.

See [`datasets.md`](datasets.md) for the synthetic generator contract.

## Explicit Pi-PLS and PLS comparison

`examples/09_pls_path_comparison.py` is the dedicated comparison example. For Pulp, Sugarcane, and
Tobacco, it writes separate Pi-PLS and ordinary-PLS component-path tables and produces one overlaid
CV-MSE figure per dataset.

This comparison is optional. Ordinary Pi-PLS use does not require fitting a PLS reference model.

## Complete Pi-PLS analyses

The real-data examples demonstrate the normal two-stage Pi-PLS workflow:

1. evaluate a Pi-PLS component path;
2. inspect the path and choose a component count;
3. read the matching predictor rank from the path table;
4. fit one fixed Pi-PLS model;
5. generate fitted-model interpretation and prediction-diagnostic artifacts.

The examples are:

- `examples/10_pulp_real_data.py`: named scalar predictors and responses, grouped-bar inspection,
  and a score-loading biplot;
- `examples/11_sugarcane_real_data.py`: wavelength-aware spectral directions, loadings, and
  coefficients;
- `examples/12_tobacco_real_data.py`: decreasing-wavenumber spectral plots, response pagination,
  and observation diagnostics.

Each dataset analysis directory contains `component_path.csv`, `component_path.pdf`, the canonical
post-analysis CSV tables, and `post_analysis.pdf`. The path PDF and the post-analysis PDF are
separate because model selection and fitted-model interpretation answer different questions.

## Example support code

The modules under `examples/_support/` implement reusable orchestration for the complete examples:

- ordinary-PLS component-path evaluation for example 09;
- CSV-to-PDF component-path plotting;
- fixed-parameter out-of-fold prediction;
- canonical post-analysis tables and report composition.

They are not required for normal estimator use. Reusable numerical analysis belongs in
`pipls.inspection`, and optional Matplotlib figures belong in `pipls.plotting`.

## Running the examples

Install the example dependencies and run all numbered examples with:

```bash
python -m pip install -e ".[examples]"
make examples
```

The complete real-data analyses are intentionally separate from `make check` because they are
application workflows and may take substantially longer than the package test suite.
