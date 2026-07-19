# Pi-PLS

`pipls` is an installable Python package for Pi-PLS, a PLS-family method for multivariate
regression. Its public interfaces follow scikit-learn conventions and provide a fixed-model
regression estimator, pipeline-aware triangular path analysis, advanced cross-validation, ordered
out-of-fold diagnostics, pure fitted-model inspection, deterministic synthetic data generation,
and transparent reference datasets.

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

For an examples-only environment, install the data-reading and plotting dependencies with:

```bash
python -m pip install -e ".[examples]"
make examples
```

## Basic model

```python
from pipls import PiPLSRegression

model = PiPLSRegression(n_components=2, predictor_rank=4)
model.fit(X_train, Y_train)
Y_pred = model.predict(X_test)
X_scores, Y_scores = model.transform(X_train, Y_train)
print(model.predictor_rank_)
print(model.decomposition_.D)
```

`PiPLSRegression` fits one explicit `(n_components, predictor_rank)` pair. It centers `X` and
`Y` during fitting and, with `scale=True`, learns their training-sample standard deviations.
`PiPLSPathCV` performs model selection by cloning and fitting fixed estimators separately inside
every training fold, then optionally refits the selected pair on the complete training set.

## Component-path analysis

Scan component counts first and write one conditional predictor-rank row per value:

```python
import pandas as pd

from pipls import PiPLSPathCV, PiPLSRegression

search = PiPLSPathCV(refit=False).fit(X_train, Y_train)

path = pd.DataFrame(search.component_path_results_)
path.to_csv("component_path.csv", index=False)
```

The default `n_components_values="all"` evaluates every admissible component count. Supply an
explicit sequence such as `[1, 2, 3, 4]` when only a subset is wanted.

After inspecting the canonical CSV, choose a component count and fit both ranks explicitly:

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
from pipls.inspection import pipls_display_factors, prediction_diagnostics

factors = pipls_display_factors(model.decomposition_)
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
test predictions. See [`docs/model_inspection.md`](docs/model_inspection.md).

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

The Pulp, Sugarcane, and Tobacco examples write separate canonical Pi-PLS and standard PLS
(NIPALS) component-path CSV files, call small imported functions to generate the comparison PDF,
and then fit a separate fixed Pi-PLS model using a visible component-count choice. The Pi-PLS CSV
always records the selected predictor rank. Tobacco uses adaptive scanning with explicit full
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
`make examples` runs every numbered example in order, including the complete Pulp, Sugarcane, and
Tobacco analyses. Examples 10–12 read the public tables directly, write separate Pi-PLS and
standard PLS component-path CSVs, generate a comparison PDF, and fit a separately chosen fixed
Pi-PLS model.

See [`docs/benchmarks.md`](docs/benchmarks.md), [`benchmarks/README.md`](benchmarks/README.md), and
[`examples/README.md`](examples/README.md).

## Documentation

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
centered-data feasibility caps. A direct `PiPLSRegression` fit warns when it has fewer than four
observations per retained predictor-rank direction.

## Repository map

- `src/pipls/`: installable package and public API;
- `docs/`: user and developer documentation;
- `examples/`: concise executable workflows;
- `datasets/`: transparent redistributable reference datasets;
- `benchmarks/`: focused package-validation plans, scripts, and ignored CSV outputs;
- `tests/`: numerical, API, integration, and repository tests;
- `.llm/`: tracked maintenance contracts for LLM-assisted development, excluded from the package.

Read [CONTRIBUTING.md](CONTRIBUTING.md) before preparing a change. For LLM-assisted maintenance,
read [`.llm/README.md`](.llm/README.md) and create a clean snapshot with `make snapshot`.
