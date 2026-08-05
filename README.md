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
retains the complete selection evidence, while `refit()` returns the fitted final estimator with
its exact `selection_`, and `oof_report()` returns ordered selection-conditioned OOF diagnostics for
that selection.

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
    rule="minimum_cv_mse",
)

Y_fitted = model.predict(X)
```

`examples/01_pulp_quick_start.py` standardizes each response and places all observed and fitted
values in one figure. These are fitted values from the final full-data model, not out-of-fold
predictions; retain the search and use `search.oof_report(..., selection=model.selection_)` when OOF diagnostics are required.

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
model = search.refit(
    X_train,
    Y_train,
    n_components=2,
)

selection = model.selection_
path = search.component_path_
rank_profile = search.predictor_rank_profile(selection.n_components)

Y_pred = model.predict(X_test)
```

For every evaluated component count, the search conditionally retains one predictor rank.
Constructor-level `predictor_rank_relative_tolerance` and
`predictor_rank_absolute_tolerance` can favor the smallest evaluated rank within a bounded
configured-score allowance; their defaults reproduce an effectively exact optimum.
`refit(..., n_components=h)` transfers the stored pair into a fitted clone without requiring the
user to copy `predictor_rank`. The returned model exposes that complete row as `model.selection_`;
use `search.predictor_rank_profile(model.selection_.n_components)` to compare the exact rank optimum
with the conditionally retained rank. `search.select(...)` remains available for selection-only work
that does not fit a final model.

The [synthetic tutorial](docs/tutorials/synthetic.md) shows the component-path and conditional
predictor-rank plots. The [path-selection reference](docs/api/path.md) and
[path-selection details](docs/path_analysis.md) cover alternative policies, splitters, and
validation protocols.

When the complete protocol is known in advance, search and final fitting can remain compact. The
Tobacco workflow demonstrates separate 10% relative tolerances for adaptive predictor-rank search
and the later minimum-CV-MSE component rule:

```python
model = PiPLSSearchCV(
    search_method="adaptive",
    predictor_rank_relative_tolerance=0.10,
).fit(
    X_train,
    Y_train,
).refit(
    X_train,
    Y_train,
    rule="minimum_cv_mse",
    relative_tolerance=0.10,
)

Y_pred = model.predict(X_test)
```

Retain the fitted search in a variable when component-path, predictor-rank-profile, or candidate
inspection is needed. `refit()` returns a fitted estimator or pipeline and does not attach it to the
search object. Modeling completes before the retained evidence and optional OOF diagnostics are
calculated:

```python
search = PiPLSSearchCV(search_method="adaptive").fit(X_train, Y_train)
model = search.refit(X_train, Y_train, rule="minimum_cv_mse")

selection = model.selection_
path = search.component_path_
rank_profile = search.predictor_rank_profile(selection.n_components)
report = search.oof_report(X_train, Y_train, selection=selection)
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
| `oof_report(X, Y, selection=...)` | Produce ordered OOF diagnostics for one existing selection |
| `pipls.inspection` | Compute immutable fitted-model and prediction diagnostics |
| Matplotlib | Optionally render those arrays with caller-controlled figures and styling |
| `pipls.datasets` | Load package-owned Pulp, Sugarcane, and Tobacco data or generate deterministic synthetic data |

Pi-PLS intentionally provides no plotting submodule: numerical inspection objects are the stable
interface, while rendering remains optional and caller-owned. Generated signatures, fitted
attributes, result shapes, and method contracts are collected in the
[API overview](docs/api/index.md).

## Examples and reference data

The package includes Pulp, Sugarcane, and Tobacco through the named
`pipls.datasets.load_pulp()`, `pipls.datasets.load_sugarcane()`, and
`pipls.datasets.load_tobacco()` loaders. Every maintained reference-data workflow uses the same
canonical package resources. There is no dataset-access extra or generic registry. The ordinary
CSV, JSON, README, and license files are also documented for direct use from R, C++, MATLAB, Julia,
or another environment in the [dataset guide](docs/datasets.md).

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
