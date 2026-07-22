# Pi-PLS

`pipls` is a Python package for Pi-PLS, a PLS-family method for multivariate regression.
Pi-PLS represents the predictive relation through paired predictor and response latent variables.
The number of components controls how many pairs are retained; predictor rank controls how much
predictor variation is available to form them.

For routine model selection, `PiPLSPathCV` evaluates component counts by cross-validation and
selects a predictor rank conditionally for each count. The user then chooses a parsimonious point
from the component path and fits one fixed `PiPLSRegression` model.

Start with:

1. [First Pi-PLS model with synthetic data](docs/tutorials/synthetic.md) for the compact selection
   and independent-test workflow.
2. [Complete Pulp analysis](docs/tutorials/pulp.md) for real-data loading, fixed-parameter OOF
   predictions, and representative model interpretation.

## Installation

From a source checkout, install the runtime package with:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

Install optional plotting or complete example dependencies when needed:

```bash
python -m pip install -e ".[plot]"
python -m pip install -e ".[examples]"
```

Development setup and repository validation commands are documented in
[CONTRIBUTING.md](CONTRIBUTING.md).

Pi-PLS supports Python 3.10 through 3.14 with NumPy `>=1.26,<3`, scikit-learn `>=1.4,<2`, and
joblib `>=1.2,<2`. See the [compatibility policy](docs/compatibility.md).

## Fit one known model

When both ranks are known, fit the fixed estimator directly:

```python
from pipls import PiPLSRegression

model = PiPLSRegression(
    n_components=2,
    predictor_rank=4,
).fit(X_train, Y_train)

Y_pred = model.predict(X_test)
```

This performs no parameter selection. The estimator learns centering and optional scaling from the
training observations and follows the ordinary scikit-learn `fit()` and `predict()` pattern. See
the [fixed-regression reference](docs/api/regression.md) for the complete contract.

## Select a model from the component path

When the ranks are not known, evaluate the path first:

```python
from pipls import PiPLSPathCV, PiPLSRegression

search = PiPLSPathCV(refit=False).fit(X_train, Y_train)
path = search.component_path_

selected = path.for_n_components(2)

model = PiPLSRegression(
    n_components=selected.n_components,
    predictor_rank=selected.predictor_rank,
).fit(X_train, Y_train)

Y_pred = model.predict(X_test)
```

For every evaluated component count, the default search selects the predictor rank that minimizes
mean response-standardized CV-MSE. `for_n_components()` retrieves that evaluated pair; it does not
repeat the optimization. Use `search.predictor_rank_profile(h)` to inspect all ranks evaluated at
one component count.

The [synthetic tutorial](docs/tutorials/synthetic.md) shows the component-path and conditional
predictor-rank plots. The [path-selection reference](docs/api/path.md),
[advanced path-search guide](docs/path_analysis.md), and
[cross-validation guide](docs/cross_validation.md) cover alternative policies and validation
protocols.

## Main interfaces

| Interface | Purpose |
|---|---|
| `PiPLSRegression` | Fit one fixed `(n_components, predictor_rank)` pair |
| `PiPLSPathCV` | Evaluate component counts and conditional predictor ranks |
| `component_path_` | Inspect one selected predictor rank for each component count |
| `predictor_rank_profile(h)` | Inspect all evaluated predictor ranks at one count |
| `pipls.inspection` | Compute immutable fitted-model and prediction diagnostics |
| `pipls.plotting` | Render optional one-axis Matplotlib figures |
| `pipls.datasets` | Generate deterministic synthetic Pi-PLS data |

Generated signatures, fitted attributes, result shapes, and method contracts are collected in the
[API overview](docs/api/index.md).

## Examples and reference data

The repository includes maintained examples for synthetic data and for the Pulp, Sugarcane, and
Tobacco datasets. Real-data examples read `X.csv` and `Y.csv` explicitly and keep analytical
results in memory. The original sources, licenses, adaptations, and DOI links are documented in the
[dataset guide](docs/datasets.md).

See the [example catalogue](docs/examples.md) for the purpose and outputs of every numbered script.
Install the `examples` extra before running them.

## Documentation

- [Tutorial 1: synthetic selection and prediction](docs/tutorials/synthetic.md)
- [Tutorial 2: complete Pulp analysis](docs/tutorials/pulp.md)
- [API overview](docs/api/index.md)
- [Advanced path-search behavior](docs/path_analysis.md)
- [Cross-validation and OOF reporting](docs/cross_validation.md)
- [Troubleshooting](docs/troubleshooting.md)
- [Model inspection](docs/model_inspection.md)
- [Reference datasets](docs/datasets.md)
- [Theory](docs/theory.md)

The package is distributed under the [BSD 3-Clause License](LICENSE).
