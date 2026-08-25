# Π-PLS

[![Tests](https://github.com/stefanlindstroem/pipls/actions/workflows/tests.yml/badge.svg)](https://github.com/stefanlindstroem/pipls/actions/workflows/tests.yml)
[![Documentation](https://github.com/stefanlindstroem/pipls/actions/workflows/documentation.yml/badge.svg)](https://github.com/stefanlindstroem/pipls/actions/workflows/documentation.yml)

`pipls` is the Python package for Π-PLS, a PLS-family method for multivariate regression.
For routine modeling, use it much like ordinary PLS: treat `n_components` as the main model-
complexity parameter, evaluate prediction error across component counts, select a count, fit, and
predict. The resulting cross-validated error table or curve is the **component path**.

Under the hood, Π-PLS also resolves a retained predictor-subspace dimension, `predictor_rank`, for
each component count. Most workflows do not need to tune that quantity separately. Advanced users
can inspect, restrict, or fix it when statistical support or scientific interpretation motivates
more direct control.

Π-PLS represents the predictive relation through paired latent modes. Each retained mode combines
one orthonormal predictor direction, one orthonormal response direction, and one nonnegative
dilation. The predictor rank controls the subspace from which those paired modes are estimated.
Ordinary PLS and other multivariate methods remain appropriate alternatives; model choice should be
based on the data and the validation design.

For routine model selection, `PiPLSSearchCV` evaluates the component path by cross-validation and
resolves one predictor rank conditionally for each count. Evidence-retaining workflows inspect the
path, create one immutable selection, optionally inspect its conditional rank evidence, diagnose
the accepted selection with OOF predictions when useful, and pass the same object to final
refitting. Compact workflows may instead apply a named rule directly
through `search.refit(X, Y, rule=...)`. The fitted model records the exact row as `selection_`, while
the search retains the complete path and split evidence.

By default, predictor rank is optimized over the complete fold-feasible integer domain and candidate
coverage is exhaustive. The EPV-inspired rule is not a hidden bound on that search: request it
explicitly with `predictor_rank_values="epv"`. Its default $c=10$ is controlled by
`samples_per_predictor_rank`; explicit rank sequences and an integer `max_predictor_rank` remain
available when the user intends to restrict the rank domain. Use `search_method="adaptive"`
explicitly when reduced candidate coverage is an acceptable computational approximation.

Programming users can also choose how the intermediate response subspace is constructed.
`response_subspace="cross_covariance"` is the package default and the construction used in the
peer-reviewed companion publication. `response_subspace="least_squares"` is an optional
least-squares/RRR-inspired software extension; configure it on `PiPLSRegression` or on the estimator
template supplied to `PiPLSSearchCV`. The search does not treat this choice as an additional search
dimension. See the [theory](docs/theory.md#response-subspace-selection) and
[`PiPLSRegression` reference](docs/api/regression.md#pipls.PiPLSRegression).

The [rendered documentation](https://stefanlindstroem.github.io/pipls/) is the primary user guide.
The source links below remain useful in a local checkout.

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

Π-PLS supports Python 3.10 through 3.14 with NumPy `>=1.26,<3`, scikit-learn `>=1.4,<2`, and
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
predictions. When OOF diagnostics are required, retain the search, create the selection explicitly,
and pass that same object to both `oof_report()` and `refit()` as shown below.

## Fit one exact model

When an advanced workflow has chosen both $h$ and $r_\pi$, fit that exact pair directly:

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
the [`PiPLSRegression` reference](docs/api/regression.md) for the complete contract.

## Select a model from the component path

When the ranks are not known, evaluate the path first:

```python
from pipls import PiPLSSearchCV

search = PiPLSSearchCV().fit(X_train, Y_train)
path = search.component_path_
# Inspect the path before choosing a component count.
selection = search.select(n_components=2)
rank_profile = search.predictor_rank_profile(selection.n_components)
# Reinspect the path with the selected row and inspect the conditional rank profile.

model = search.refit(
    X_train,
    Y_train,
    selection=selection,
)
Y_pred = model.predict(X_test)
```

For every evaluated component count, the search conditionally retains one predictor rank.
Constructor-level `predictor_rank_relative_tolerance` and
`predictor_rank_absolute_tolerance` can favor the smallest evaluated rank within a bounded
configured-score allowance; their defaults reproduce an effectively exact optimum.
`search.select(n_components=h)` returns the complete stored pair without fitting and avoids asking
the user to copy `predictor_rank`. The same immutable object can then configure OOF reporting and
final refitting. After a successful fit, `model.selection_` is that exact selection and can be used
to verify fitted-model provenance. Rule-based and component-count refitting remain available for
compact workflows that do not need to retain an earlier selection.

The [synthetic tutorial](docs/tutorials/synthetic.md) shows the component-path and conditional
predictor-rank plots. The [`PiPLSSearchCV` reference](docs/api/path.md), [Path and selection](docs/path_selection.md),
and [OOF diagnostics](docs/oof_diagnostics.md) define alternative rank policies, splitters,
selection rules, selection-conditioned OOF semantics, and the main computational consequences of
candidate coverage and validation splits.

When the complete protocol is known in advance, search and final fitting can remain compact. The
same separate 10% relative tolerances used in the Tobacco analysis can be written as one automatic
search-and-refit expression:

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

Retain the fitted search in a variable when component-path, predictor-rank-profile, candidate, or
OOF diagnostic provenance matters. In that route, inspect the component path first, create one
selection, and review its conditional evidence before final fitting so every operation refers to
the same stored row:

```python
search = PiPLSSearchCV(search_method="adaptive").fit(X_train, Y_train)
path = search.component_path_
# Inspect path before choosing a component-count rule or tolerance.
selection = search.select(
    rule="minimum_cv_mse",
    relative_tolerance=0.10,
)
rank_profile = search.predictor_rank_profile(selection.n_components)
report = search.oof_report(X_train, Y_train, selection=selection)
model = search.refit(X_train, Y_train, selection=selection)
```

`refit()` returns a fitted estimator or pipeline and does not attach it to the search object.
`model.selection_` records the exact supplied selection after fitting succeeds; it is fitted-model
provenance rather than the handoff used to select or review the row.

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

Π-PLS intentionally provides no plotting submodule: numerical inspection objects are the stable
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

- [Rendered documentation](https://stefanlindstroem.github.io/pipls/)
- Source documentation:
  - [Tutorial 1: Pulp quick start](docs/tutorials/quick_start.md)
  - [Tutorial 2: synthetic path inspection](docs/tutorials/synthetic.md)
  - [Tutorial 3: complete Pulp analysis](docs/tutorials/pulp.md)
  - [API overview](docs/api/index.md)
  - [Path and selection](docs/path_selection.md)
  - [OOF diagnostics](docs/oof_diagnostics.md)
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
