# Π-PLS

[![Tests](https://github.com/stefanlindstroem/pipls/actions/workflows/tests.yml/badge.svg)](https://github.com/stefanlindstroem/pipls/actions/workflows/tests.yml)
[![Documentation](https://github.com/stefanlindstroem/pipls/actions/workflows/documentation.yml/badge.svg)](https://github.com/stefanlindstroem/pipls/actions/workflows/documentation.yml)

`pipls` is the Python package for panoramic partial least squares (Π-PLS), a PLS-family method for
multivariate regression. Π-PLS represents the predictive relation through paired latent modes: each
retained mode contains one orthonormal predictor direction, one orthonormal response direction, and
one nonnegative dilation.

For routine modeling, `n_components` is the main model-complexity parameter. `PiPLSSearchCV`
evaluates prediction error across component counts and conditionally resolves one retained predictor
rank for each count. Advanced workflows can inspect or restrict that predictor-rank search, or fit an
explicit component-count/predictor-rank pair with `PiPLSRegression`.

The [rendered documentation](https://stefanlindstroem.github.io/pipls/) is the primary user guide.

## Installation

From a source checkout, create an environment and install the runtime package with:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install .
```

Install the plotting dependencies used by the numbered examples and tutorials when needed:

```bash
python -m pip install ".[examples]"
```

Π-PLS supports Python 3.10 through 3.14 with NumPy `>=1.26,<3`, scikit-learn `>=1.4,<2`, and
joblib `>=1.2,<2`. See the [compatibility policy](docs/compatibility.md). Contributor setup and
repository validation commands are documented in [CONTRIBUTING.md](CONTRIBUTING.md).

## Quick start

The package includes the multivariate Pulp dataset, so a complete component-path search and final
fit requires no external files:

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

`Y_fitted` contains fitted values from the final full-data model. For selection-conditioned
out-of-fold diagnostics, retain the search and create one selection explicitly:

```python
search = PiPLSSearchCV().fit(X, Y)
selection = search.select(rule="minimum_cv_mse")
report = search.oof_report(X, Y, selection=selection)
model = search.refit(X, Y, selection=selection)
```

The same immutable selection therefore identifies the row used for OOF diagnostics and final
refitting. Independent post-selection performance assessment still requires nested cross-validation
or an independent test set; see [OOF diagnostics](docs/oof_diagnostics.md).

## Fit one exact model

When both the paired-mode count $h$ and retained predictor rank $r_\pi$ are already known, fit that
pair directly:

```python
from pipls import PiPLSRegression

model = PiPLSRegression(
    n_components=2,
    predictor_rank=4,
).fit(X_train, Y_train)

Y_pred = model.predict(X_test)
```

This route performs no parameter selection. See the
[`PiPLSRegression` reference](docs/api/regression.md) for the estimator contract and
[Theory](docs/theory.md) for the mathematical construction.

## Where to go next

- [Quick start with Pulp](docs/tutorials/quick_start.md) develops the installed-data example and its
  prediction diagnostics.
- [Inspect and select with synthetic data](docs/tutorials/synthetic.md) introduces component-path
  inspection, conditional predictor-rank evidence, and independent-test prediction.
- [Complete Pulp analysis](docs/tutorials/pulp.md) covers selection-conditioned OOF diagnostics,
  final refitting, and model interpretation.
- [Path and selection](docs/path_selection.md) defines predictor-rank policies, component selection,
  cross-validation metadata, and computational trade-offs.
- [Model inspection](docs/model_inspection.md) describes the immutable numerical results used for
  latent-structure, coefficient, and prediction diagnostics.
- [API overview](docs/api/index.md) collects the public estimator, search, dataset, and inspection
  interfaces.

The package includes Pulp, Sugarcane, and Tobacco through `pipls.datasets`. Their provenance,
licenses, and language-neutral raw-resource layout are documented in the
[dataset guide](docs/datasets.md). Numbered executable workflows are summarized in the
[example catalogue](docs/examples.md).

## Authors, license, and citation

The code and repository-authored documentation are copyright (c) 2026 Vishal Agrawal,
Fritjof Nilsson, and Stefan B. Lindström and are distributed under the
[BSD 3-Clause License](LICENSE). Included reference datasets retain their own license and
attribution notices.

The companion paper is under revision:

> Agrawal, V., Nilsson, F., and Lindström, S. B. (2026). Panoramic Partial Least Squares
> (Pi-PLS): Transparent, parsimonious, and more interpretable multivariate regression model.
> Manuscript under revision at *Computers & Chemical Engineering*, manuscript CACE-D-26-00847.

See [authors, license, and citation](docs/citation.md) for the full scope and
[`CITATION.cff`](CITATION.cff) for machine-readable citation metadata.
