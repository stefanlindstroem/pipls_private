# Pi-PLS

`pipls` is an installable Python package for Pi-PLS, a PLS-family method for multivariate
regression. Pi-PLS represents the predictive relation through paired predictor and response latent
variables. The number of components controls how many pairs are retained, while predictor rank
controls how much predictor variation is available to form them.

In routine use, `PiPLSPathCV` scans component counts by cross-validation and reports
cross-validated mean squared error (CV-MSE) for each count. The user then chooses a parsimonious
point, often an elbow or plateau in the CV-MSE curve, and fits one fixed `PiPLSRegression` model.
The public interfaces follow scikit-learn conventions and also provide validation diagnostics, fitted-model inspection, synthetic data, and
transparent reference datasets. See the [theory overview](docs/theory.md) for the construction.

The recommended documentation entry point is the
[complete Pulp tutorial](docs/tutorials/pulp.md), which covers pipeline construction, parameter
selection, fixed fitting, out-of-fold prediction, and figure-by-figure interpretation.

## Installation

For development from a source checkout:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
make check
```

For a later terminal session, reactivate the existing environment before running commands:

```bash
cd /path/to/pipls
source .venv/bin/activate
git status
make check
```

After pulling a change that adds dependencies, refresh the active development environment with:

```bash
python -m pip install -e ".[dev]"
```

The Makefile is the command index for repository work. Run:

```bash
make help
```

to list the maintained targets and their purpose.

For an examples environment, install the data-reading and plotting dependencies with:

```bash
python -m pip install -e ".[examples]"
PYTHONPATH=src MPLBACKEND=Agg python examples/01_minimal_fit_and_plot.py
```

`make examples` runs every numbered example, including the complete and slower real-data analyses.

Pi-PLS supports Python 3.10 through 3.14 with NumPy `>=1.26,<3`, scikit-learn
`>=1.4,<2`, and joblib `>=1.2,<2`. See the
[compatibility guide](docs/compatibility.md) for the minimum-dependency and CI policy.

## Quickstart: one fixed model

The shortest complete workflow uses literal NumPy matrices, one fixed fit, and one plot. It performs
no cross-validation or parameter selection:

```python
import matplotlib.pyplot as plt
import numpy as np

from pipls import PiPLSRegression
from pipls.inspection import pipls_display_factors
from pipls.plotting import (
    plot_pipls_dilation,
    plot_pipls_predictor_directions,
    plot_pipls_response_directions,
    plot_pipls_weighted_response_directions,
)

X = np.array(
    [
        [1.0, 2.0, 0.5],
        [2.0, 1.0, 1.0],
        [3.0, 4.0, 1.5],
        [4.0, 3.0, 2.0],
        [5.0, 6.0, 2.5],
        [6.0, 5.0, 3.0],
        [7.0, 8.0, 3.5],
        [8.0, 7.0, 4.0],
    ]
)
Y = np.array(
    [
        [1.2, 2.0],
        [1.8, 1.7],
        [3.1, 3.3],
        [3.7, 3.0],
        [5.2, 4.6],
        [5.8, 4.3],
        [7.1, 5.9],
        [7.7, 5.6],
    ]
)

model = PiPLSRegression(n_components=1, predictor_rank=2).fit(X, Y)
Y_fitted = model.predict(X)

factors = pipls_display_factors(model.decomposition_)
figure, axes = plt.subplots(2, 2, figsize=(11, 8), layout="constrained")
plot_pipls_predictor_directions(
    factors,
    predictor_style="bar",
    predictor_names=["Temperature", "Pressure", "Flow rate"],
    ax=axes[0, 0],
)
plot_pipls_dilation(factors, ax=axes[0, 1])
plot_pipls_response_directions(
    factors,
    response_names=["Yield", "Purity"],
    ax=axes[1, 0],
)
plot_pipls_weighted_response_directions(
    factors,
    response_names=["Yield", "Purity"],
    ax=axes[1, 1],
)
for axis in (axes[0, 0], axes[1, 0], axes[1, 1]):
    axis.legend(title="Component")
figure.suptitle("Minimal Pi-PLS fit")
```

`PiPLSRegression` fits one explicit `(n_components, predictor_rank)` pair. It centers `X` and
`Y` during fitting and, with `scale=True`, learns their training-sample standard deviations.
Fitting, numerical inspection, and plotting remain separate operations. See the runnable
[`01_minimal_fit_and_plot.py`](examples/01_minimal_fit_and_plot.py) example and the
[quickstart guide](docs/quickstart.md).

`PiPLSPathCV` is the separate model-selection interface. It clones and fits fixed estimators inside
every training fold, then optionally refits the selected pair on the complete training set.

## Choosing latent components

A component path is the table or curve obtained by scanning component counts. Each row contains the
mean CV-MSE and the predictor rank selected for that component count:

```python
import pandas as pd

from pipls import PiPLSPathCV, PiPLSRegression

search = PiPLSPathCV(refit=False).fit(X_train, Y_train)

path = pd.DataFrame(search.component_path_results_)
path.to_csv("component_path.csv", index=False)
```

The default `n_components_values="all"` evaluates every admissible component count. Supply an
explicit sequence such as `[1, 2, 3, 4]` when only a subset is wanted.

Plot or inspect CV-MSE against component count, choose an elbow, plateau, or other justified
point, and fit both ranks explicitly:

```python
chosen_n_components = 3
path = pd.read_csv("component_path.csv").set_index("n_components")
chosen_predictor_rank = int(path.loc[chosen_n_components, "predictor_rank"])
model = PiPLSRegression(
    n_components=chosen_n_components,
    predictor_rank=chosen_predictor_rank,
).fit(X_train, Y_train)
```

`best_params_` identifies the best evaluated pair under the configured scorer. With the default
negative response-standardized MSE scorer, this is the evaluated pair with the smallest mean
CV-MSE. Adaptive search may leave admissible pairs unevaluated, and the examples still present
component-count selection as a user decision. For explicit validation reporting:

```python
from sklearn.model_selection import LeaveOneOut

search = PiPLSPathCV(
    cv=LeaveOneOut(),
    return_oof_predictions=True,
).fit(X_train, Y_train)
print(search.validation_report_)
print(search.oof_predictions_)
```

## Numerical model inspection

Inspect a fitted Pi-PLS factorization without changing the estimator:

```python
from pipls.inspection import (
    biplot_coordinates,
    latent_structure,
    pipls_display_factors,
    prediction_diagnostics,
)

factors = pipls_display_factors(model.decomposition_)
structure = latent_structure(model)
biplot = biplot_coordinates(structure, components=(0, 1))

Y_pred = model.predict(X_test)
diagnostics = prediction_diagnostics(
    Y_test,
    Y_pred,
    prediction_kind="external test predictions",
)
```

The display factors are read-only copies of $P$, $D$, and $Q$ with deterministic component signs;
they preserve $P D Q^\mathsf{T}$. Prediction diagnostics use residuals $y-\hat y$ and standardize
all responses from the supplied observed-response center and sample standard deviation. The
provenance label distinguishes fitted, fixed-parameter OOF, selection-conditioned OOF, and external
test predictions.

Variable labels are supplied separately from numerical inspection. For a header-bearing CSV
workflow, read them visibly at the example boundary:

```python
import pandas as pd

X = pd.read_csv("X.csv")
Y = pd.read_csv("Y.csv")
predictor_names = X.columns.astype(str).tolist()
response_names = Y.columns.astype(str).tolist()
```

Users working with NumPy arrays can supply the same lists from any explicit metadata source.
Install the optional plotting dependency and render the computed results explicitly:

```python
import matplotlib.pyplot as plt

from pipls.plotting import (
    plot_biplot,
    plot_pipls_dilation,
    plot_pipls_predictor_directions,
    plot_pipls_response_directions,
    plot_pipls_weighted_response_directions,
    plot_observed_vs_predicted,
    plot_residuals_vs_predicted,
    plot_standardized_rmse,
    plot_scores,
    plot_x_loadings,
)

factor_figure, factor_axes = plt.subplots(2, 2, figsize=(12, 9), layout="constrained")
plot_pipls_predictor_directions(
    factors,
    predictor_style="bar",
    predictor_names=predictor_names,
    ax=factor_axes[0, 0],
)
plot_pipls_dilation(factors, ax=factor_axes[0, 1])
plot_pipls_response_directions(
    factors,
    response_names=response_names,
    ax=factor_axes[1, 0],
)
plot_pipls_weighted_response_directions(
    factors,
    response_names=response_names,
    ax=factor_axes[1, 1],
)
for axis in (factor_axes[0, 0], factor_axes[1, 0], factor_axes[1, 1]):
    axis.legend(title="Component")
prediction_figure, prediction_axes = plt.subplots(
    1, 3, figsize=(13, 4.2), layout="constrained"
)
plot_observed_vs_predicted(
    diagnostics,
    response_names=response_names,
    include_prediction_kind=False,
    ax=prediction_axes[0],
)
plot_residuals_vs_predicted(
    diagnostics,
    response_names=response_names,
    include_prediction_kind=False,
    ax=prediction_axes[1],
)
plot_standardized_rmse(
    diagnostics,
    response_names=response_names,
    include_prediction_kind=False,
    ax=prediction_axes[2],
)
prediction_axes[0].legend()
prediction_axes[1].legend()
prediction_figure.suptitle(diagnostics.prediction_kind)
score_figure, score_axis = plot_scores(structure, components=(0, 1))
biplot_figure, biplot_axis = plot_biplot(
    biplot,
    predictor_names=predictor_names,
)
loading_figure, loading_axis = plot_x_loadings(
    structure,
    predictor_style="bar",
    predictor_names=predictor_names,
)
loading_axis.legend(title="Component")
```

Every plotting function returns `(figure, axis)` and accepts an optional caller-supplied `ax`, so
the same chart can stand alone or draw into a user-owned panel. Package plotters do not create
panels, legends, figure-level titles, files, displays, or closing operations. Multi-series artists
are labeled, but callers add and position legends themselves. The shared
PLS-family score, balanced score-loading biplot, X/Y-loading, coefficient, and raw
observation-diagnostic figures are available from `pipls.plotting`. The numerical extraction
accepts compatible fitted `PiPLSRegression` and scikit-learn `PLSRegression` models. Use
`predictor_style="line"` with an explicit physical coordinate and axis label for spectra. Install
with `python -m pip install "pipls[plot]"`.
See [`docs/model_inspection.md`](docs/model_inspection.md) and the complete
[Pulp post-analysis example](examples/10_pulp_real_data.py).

## Synthetic data

```python
from pipls.datasets import make_pipls_train_test

train, test = make_pipls_train_test(
    n_train=120,
    n_test=40,
    n_features=20,
    n_targets=5,
    n_shared=2,
    n_predictor_specific=2,
    n_response_specific=1,
    random_state=0,
)

model.fit(train.X, train.Y)
print(model.score(test.X, test.Y))
```

## Reference datasets and examples

The repository includes transparent examples for pulp, sugarcane, and tobacco. Each example
reads comma-delimited `X.csv` and `Y.csv` files directly with pandas and shows all analysis-facing
matrix construction in ordinary user code.

```python
import pandas as pd

from pipls import PiPLSPathCV

X = pd.read_csv("datasets/pulp/X.csv")
Y = pd.read_csv("datasets/pulp/Y.csv")
search = PiPLSPathCV(refit=False).fit(X, Y)
path = pd.DataFrame(search.component_path_results_)
```

Example 09 writes separate canonical Pi-PLS and standard PLS (NIPALS) component-path CSV
files for Pulp, Sugarcane, and Tobacco and derives the shared CV-MSE comparison figures from those
tables. Examples 10–12 then perform Pi-PLS-only analyses. Each writes one Pi-PLS path, fits one
selected Pi-PLS model, and uses that model for selection-conditioned OOF predictions and all shared
score, loading, coefficient, biplot, and observation analyses. Seven common canonical post-analysis
CSV files are rebuilt into multipage reports whose figures are composed entirely in the example
layer. Pulp groups scores, its balanced two-component biplot, X loadings, and Y loadings in one
$2\times2$ page. Sugarcane uses a $1\times3$ scores/loading page. Tobacco adds an eighth table with
raw score-distance and X-reconstruction-residual diagnostics, places those diagnostics with scores
and loadings in a $2\times2$ page, preserves the decreasing wavenumber axis from `X.csv`, and
paginates all thirteen responses in source order. Every Pi-PLS
path CSV records the selected predictor rank. Tobacco uses adaptive scanning with explicit full
predictor SVD; randomized-SVD behavior is covered by the solver-consistency benchmark. Install the
`examples` extra to run them. See [`examples/README.md`](examples/README.md) and
[`datasets/README.md`](datasets/README.md).

## Lightweight benchmarks

The repository contains four focused synthetic validation benchmarks. Each benchmark owns one
scientific question and one minimal CSV output:

- fixed-structure recovery;
- adaptive rank selection;
- predictor-nuisance comparison with ordinary PLS;
- full-versus-randomized SVD consistency.

The real-data workflows are explicit application runs rather than benchmark or test-suite jobs.
`make examples` runs every numbered example in order. Example 09 owns the Pulp, Sugarcane, and
Tobacco Pi-PLS-versus-PLS CV-MSE comparisons. Examples 10–12 are normal Pi-PLS analyses: each
writes one Pi-PLS `component_path.csv` and `component_path.pdf` beside its separate
`post_analysis.pdf`, then fits and interprets one selected Pi-PLS model.

See [`docs/benchmarks.md`](docs/benchmarks.md), [`benchmarks/README.md`](benchmarks/README.md), and
[`examples/README.md`](examples/README.md).

## Documentation

Install the documentation dependencies once, then either build or preview the site:

```bash
python -m pip install -e ".[docs]"
make docs
make docs-serve
```

`make docs-serve` prints and serves the documentation at `http://127.0.0.1:8000/`; stop the local
server with `Ctrl+C`. The strict `make docs` build writes the generated site to `site/`. To verify
that the same site can
be built from the shipped source distribution, run:

```bash
make docs-dist
```

This creates a temporary clean environment and does not retain generated files in the repository.

To validate the installable runtime artifacts themselves, run:

```bash
make dist-check
```

This builds the wheel and source distribution once, installs each into its own clean virtual
environment outside the checkout, and runs the same public-import, metadata, fit, and prediction
smoke test against both artifacts.

- [Complete Pulp tutorial](docs/tutorials/pulp.md)
- [Quickstart](docs/quickstart.md)
- [Examples](docs/examples.md)
- [Documentation index](docs/index.md)
- [Estimator API](docs/estimator_api.md)
- [Parameter selection](docs/parameter_selection.md)
- [Path analysis](docs/path_analysis.md)
- [Cross-validation and OOF reporting](docs/cross_validation.md)
- [Numerical model inspection](docs/model_inspection.md)
- [Model-internal preprocessing](docs/preprocessing.md)
- [Datasets and synthetic generation](docs/datasets.md)
- [Lightweight validation benchmarks](docs/benchmarks.md)
- [Theory](docs/theory.md)
- [Reproducibility and validation](docs/reproducibility.md)

The estimator follows scikit-learn and `PLSRegression` conventions for coefficient orientation,
latent-score transforms, feature names, pandas output containers, fitted weights/loadings, and
`random_state` values. Pi-PLS-specific factorization output and numerical diagnostics are grouped
in the public read-only `decomposition_` result rather than duplicated as top-level aliases.

The default selection scorer is the ordinary callable
`pipls.metrics.neg_response_standardized_mean_squared_error`; explicit scikit-learn scorer names,
other callables, and `None` remain supported.

`PiPLSPathCV` defaults to `samples_per_predictor_rank=5` and `cv=5`. Its support term uses the
total number of observations supplied to `fit()`; cross-validation training folds impose
centered-data feasibility caps. A direct `PiPLSRegression` fit warns when it has fewer than three
observations per retained predictor-rank direction.

## Repository map

- `src/pipls/`: installable package and public API;
- `docs/`: user and developer documentation;
- `examples/`: numbered user workflows plus underscore-prefixed support for complete analyses;
- `datasets/`: transparent redistributable reference datasets;
- `benchmarks/`: focused package-validation plans, scripts, and ignored CSV outputs;
- `tests/`: numerical, API, integration, and repository tests;
- `.llm/`: tracked maintenance contracts for LLM-assisted development, excluded from the package.

Read [CONTRIBUTING.md](CONTRIBUTING.md) before preparing a change. For LLM-assisted maintenance,
read [`.llm/README.md`](.llm/README.md) and create a clean snapshot with `make snapshot`.
