# Pi-PLS

`pipls` is a Python package for Pi-PLS, a PLS-family method for multivariate regression.
Pi-PLS represents the predictive relation through paired latent modes. Each mode combines one
orthonormal predictor direction, one orthonormal response direction, and one nonnegative dilation.
Public `n_components` counts those paired modes; `predictor_rank` controls the dimension of the
retained predictor subspace from which they are estimated.

Pi-PLS is intended for problems with several responses where the predictor block may contain
structured variation that is not equally useful for prediction. Its two rank controls let users
examine the predictor subspace and the paired predictive relation separately. This does not make
Pi-PLS preferable for every regression problem; ordinary PLS and other multivariate methods remain
appropriate alternatives whose suitability depends on the data and validation design.

For routine model selection, `PiPLSSearchCV` evaluates component counts by cross-validation and
selects a predictor rank conditionally for each count. Users may inspect the path and fit one fixed
model manually, or apply a named post-fit rule through `search.refit(X, Y, rule=...)`. The search
retains the complete selection evidence, while `refit()` returns the fitted final estimator and
`validation_report()` returns ordered selection-conditioned OOF diagnostics for one selected row.

The rendered documentation is the primary user guide. On GitHub, open the latest
[`github-pages` deployment](../../deployments/github-pages). The source links below remain useful
in a local checkout.

Start with:

1. [Quick start with Pulp](docs/tutorials/quick_start.md) for one installed real-data fit and one
   observed-versus-fitted plot.
2. [Inspect and select with synthetic data](docs/tutorials/synthetic.md) for component-path
   inspection and independent-test prediction.
3. [Complete Pulp analysis](docs/tutorials/pulp.md) for selection-conditioned OOF predictions and
   representative model interpretation.

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

## Quick start with Pulp

The package includes the multivariate Pulp dataset, so the shortest complete search and final fit
requires no external files:

```python
from pipls import PiPLSSearchCV
from pipls.datasets import load_pulp

X, Y = load_pulp(return_X_y=True)

model = PiPLSSearchCV().fit(X, Y).refit(
    X,
    Y,
    rule="one_standard_error",
)

Y_fitted = model.predict(X)
```

`examples/01_pulp_quick_start.py` standardizes each response and places all observed and fitted
values in one figure. These are fitted values from the final full-data model, not out-of-fold
predictions; use `search.validation_report(...)` when predictive validation is required.

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
from pipls import PiPLSSearchCV

search = PiPLSSearchCV().fit(X_train, Y_train)
path = search.component_path_

# Inspect path.cv_mse_mean and search.predictor_rank_profile(2).
model = search.refit(
    X_train,
    Y_train,
    n_components=2,
)

Y_pred = model.predict(X_test)
```

For every evaluated component count, the default search selects the predictor rank that minimizes
mean response-standardized CV-MSE. `refit(..., n_components=h)` transfers that stored pair into a fitted clone without requiring
the user to copy `predictor_rank`. Use `search.select(n_components=h)` when the scalar row
itself is needed for annotation or reporting, and use `search.predictor_rank_profile(h)` to
inspect all ranks evaluated at one component count.

The [synthetic tutorial](docs/tutorials/synthetic.md) shows the component-path and conditional
predictor-rank plots. The [path-selection reference](docs/api/path.md) and
[path-selection details](docs/path_analysis.md) cover alternative policies, splitters, and
validation protocols.

When the complete protocol is known in advance, search and final fitting can remain compact. This
example uses adaptive predictor-rank search and the 1-SE component rule:

```python
model = PiPLSSearchCV(search_method="auto").fit(
    X_train,
    Y_train,
).refit(
    X_train,
    Y_train,
    rule="one_standard_error",
)

Y_pred = model.predict(X_test)
```

Retain the fitted search in a variable when component-path, predictor-rank-profile, or candidate
inspection is needed. `refit()` returns a fitted estimator or pipeline and does not attach it to the
search object. Selection-conditioned OOF diagnostics are requested explicitly and reuse the exact
validation splits materialized by `fit()`:

```python
report = search.validation_report(
    X_train,
    Y_train,
    n_components=2,
)
```

The caller must pass the same observations in the same row order; the search retains split indices,
not the training matrices.

## Main interfaces

| Interface | Purpose |
|---|---|
| `PiPLSRegression` | Fit one fixed paired-mode count and retained predictor-subspace dimension |
| `PiPLSSearchCV` | Evaluate the path, inspect evidence, refit one row, or validate one row explicitly |
| `component_path_` | Inspect one conditionally chosen predictor rank for each paired-mode count |
| `predictor_rank_profile(h)` | Inspect all evaluated predictor ranks at one paired-mode count |
| `validation_report(X, Y, ...)` | Produce ordered OOF diagnostics for one stored path row |
| `pipls.inspection` | Compute immutable fitted-model and prediction diagnostics |
| Matplotlib | Optionally render those arrays with caller-controlled figures and styling |
| `pipls.datasets` | Load package-owned Pulp and Sugarcane data or generate deterministic synthetic data |

Pi-PLS intentionally provides no plotting submodule: numerical inspection objects are the stable
interface, while rendering remains optional and caller-owned. Generated signatures, fitted
attributes, result shapes, and method contracts are collected in the
[API overview](docs/api/index.md).

## Examples and reference data

The package includes Pulp and Sugarcane through the named `pipls.datasets.load_pulp()` and
`pipls.datasets.load_sugarcane()` loaders. The maintained Pulp quick start, component-path
comparison, and complete tutorial use the installed Pulp dataset. Sugarcane's maintained workflows
remain on the byte-identical repository CSV copy until the consumer-migration patch; Tobacco
remains an explicit repository-CSV workflow. There is no dataset-access extra or generic registry.
Original sources, licenses, adaptations, and DOI links are documented in the
[dataset guide](docs/datasets.md).

See the [example catalogue](docs/examples.md) for the purpose and outputs of every numbered script.
Install the `examples` extra before running them.

## Documentation

- [Rendered documentation](../../deployments/github-pages)
- Source documentation:
  - [Tutorial 1: Pulp quick start](docs/tutorials/quick_start.md)
  - [Tutorial 2: synthetic path inspection](docs/tutorials/synthetic.md)
  - [Tutorial 3: complete Pulp analysis](docs/tutorials/pulp.md)
  - [API overview](docs/api/index.md)
  - [Path-selection details](docs/path_analysis.md)
  - [Troubleshooting](docs/troubleshooting.md)
  - [Model inspection](docs/model_inspection.md)
  - [Reference datasets](docs/datasets.md)
  - [Theory](docs/theory.md)
  - [Companion-manuscript synthetic data](docs/manuscript_reproduction.md)

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
