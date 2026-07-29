# Pi-PLS

`pipls` is a Python package for Pi-PLS, a PLS-family method for multivariate regression.
Pi-PLS represents the predictive relation through paired predictor and response latent variables.
The number of components controls how many pairs are retained; predictor rank controls how much
predictor variation is available to form them.

Pi-PLS is intended for problems with several responses where the predictor block may contain
structured variation that is not equally useful for prediction. Its two rank controls let users
examine the predictor subspace and the paired predictive relation separately. This does not make
Pi-PLS preferable for every regression problem; ordinary PLS and other multivariate methods remain
appropriate alternatives whose suitability depends on the data and validation design.

For routine model selection, `PiPLSSearchCV` evaluates component counts by cross-validation and
selects a predictor rank conditionally for each count. Users may inspect the path and fit one fixed
`PiPLSRegression`, or declare a final selection rule and let the search object refit that selected
pair while retaining the complete selection record.

The rendered documentation is the primary user guide. On GitHub, open the latest
[`github-pages` deployment](../../deployments/github-pages). The source links below remain useful
in a local checkout.

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
python -m pip install .
```

Install the complete example and tutorial plotting dependencies when needed:

```bash
python -m pip install ".[examples]"
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
from pipls import PiPLSRegression, PiPLSSearchCV

search = PiPLSSearchCV().fit(X_train, Y_train)
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
predictor-rank plots. The [path-selection reference](docs/api/path.md) and
[path-selection details](docs/path_analysis.md) cover alternative policies, splitters, and
validation protocols.

When the complete protocol is known in advance, selection and final fitting can be performed in one
call. This example uses adaptive predictor-rank search and the 1-SE component rule:

```python
search = PiPLSSearchCV(
    search_method="auto",
    selection_rule="one_standard_error",
    refit=True,
).fit(X_train, Y_train)

model = search.selected_pipls_
Y_pred = model.predict(X_test)
```

`best_*` still identifies the global configured-score optimum. `selected_result_`,
`selected_params_`, `validation_report_`, and `selected_pipls_` identify the model produced by the
declared protocol. A component choice made after inspecting the path should remain an explicit
two-stage workflow.

## Main interfaces

| Interface | Purpose |
|---|---|
| `PiPLSRegression` | Fit one fixed `(n_components, predictor_rank)` pair |
| `PiPLSSearchCV` | Evaluate the path and optionally refit an explicitly selected path row |
| `component_path_` | Inspect one selected predictor rank for each component count |
| `predictor_rank_profile(h)` | Inspect all evaluated predictor ranks at one count |
| `pipls.inspection` | Compute immutable fitted-model and prediction diagnostics |
| Matplotlib | Optionally render those arrays with caller-controlled figures and styling |
| `pipls.datasets` | Generate deterministic synthetic Pi-PLS data |

Pi-PLS intentionally provides no plotting submodule: numerical inspection objects are the stable
interface, while rendering remains optional and caller-owned. Generated signatures, fitted
attributes, result shapes, and method contracts are collected in the
[API overview](docs/api/index.md).

## Examples and reference data

The repository includes maintained examples for synthetic data and for the Pulp, Sugarcane, and
Tobacco datasets. Real-data examples read `X.csv` and `Y.csv` explicitly and keep analytical
results in memory. The CSV assets are ordinary repository files; there is no dataset-access extra,
registry, or package-owned loader. The original sources, licenses, adaptations, and DOI links are
documented in the [dataset guide](docs/datasets.md).

See the [example catalogue](docs/examples.md) for the purpose and outputs of every numbered script.
Install the `examples` extra before running them.

## Documentation

- [Rendered documentation](../../deployments/github-pages)
- Source documentation:
  - [Tutorial 1: synthetic selection and prediction](docs/tutorials/synthetic.md)
  - [Tutorial 2: complete Pulp analysis](docs/tutorials/pulp.md)
  - [API overview](docs/api/index.md)
  - [Path-selection details](docs/path_analysis.md)
  - [Troubleshooting](docs/troubleshooting.md)
  - [Model inspection](docs/model_inspection.md)
  - [Reference datasets](docs/datasets.md)
  - [Theory](docs/theory.md)

## Authors, license, and citation

The code and repository-authored documentation are copyright (c) 2026 Vishal Agrawal,
Fritjof Nilsson, and Stefan B. Lindström. They are distributed under the
[BSD 3-Clause License](LICENSE), which permits commercial use, redistribution, and modification
provided its conditions are followed, including retention of the copyright notice, conditions, and
disclaimer. Included reference datasets retain their own license and attribution notices.

The companion paper is under revision:

> Agrawal, V., Nilsson, F., and Lindström, S. B. (2026). Panoramic Partial Least Squares
> (Pi-PLS): Transparent, parsimonious, and more interpretable multivariate regression model.
> Manuscript under revision at *Computers & Chemical Engineering*, manuscript
> CACE-D-26-00847.

See [authors, license, and citation](docs/citation.md) for the full scope and
[`CITATION.cff`](CITATION.cff) for machine-readable citation metadata.
