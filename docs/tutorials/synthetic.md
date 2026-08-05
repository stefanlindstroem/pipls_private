# Inspect and refit a manually selected Pi-PLS model

This tutorial expands the [Pulp quick start](quick_start.md) by retaining the fitted search object,
inspecting its component path, creating one explicit selection, and passing that same immutable
selection to the final full-data refit. Deterministic synthetic training and test data make the
latent structure known and keep prediction assessment independent of model selection. For Pi-PLS,
`n_components` is the number of paired latent modes.

The workflow is to generate independent training and test data, search the candidate models, inspect
the search evidence, create one selection, refit that exact selection, predict the external test
data, and render the completed results.

```mermaid
flowchart TD
    A[Fit search at dummy number of components]
    B[Inspect component path]
    C[Identify suitable number of components h*]
    D[Set chosen component count]
    E[Create selection with n_components = h*]
    F[Inspect predictor-rank profile]
    G[Refit using selection]
    H[Predict independent test data]

    A --> B --> C
    C -. go back and set value .-> D
    D --> E --> F --> G --> H
```

## What this tutorial covers

You will:

1. generate independent training and test observations;
2. evaluate the component and predictor-rank search;
3. inspect the component path and create one manual selection;
4. inspect the conditional predictor-rank profile for that selection;
5. refit the exact selected pair on all training observations;
6. predict the independent test responses and render the numerical results.

The tutorial deliberately stops after one prediction plot. Scores, loadings, Pi-PLS factorization
plots, and selection-conditioned OOF diagnostics are introduced in the
[complete Pulp tutorial](pulp.md).

## Define the validation splitter

The component path uses reproducible shuffled five-fold cross-validation. The splitter definition
is shown explicitly because it determines every fold-level fit and every CV-MSE value below:

```python
--8<-- "examples/02_synthetic_path_selection.py:import-synthetic-kfold"
--8<-- "examples/02_synthetic_path_selection.py:define-synthetic-cv"
```

For grouped, blocked, or ordered observations, replace `KFold` with a splitter that represents the
sampling design rather than shuffling those structures.

## Generate training and test data

The generator creates two independent sample blocks from one latent model:

```python
--8<-- "examples/02_synthetic_path_selection.py:generate-synthetic-data"
```

The generating structure contains:

- two shared directions that affect both predictors and responses;
- two predictor-specific directions that affect only the predictors;
- one response-specific direction that affects only the responses.

The shared dimension, and therefore the intended predictive paired-mode count, is two, while the
predictor block contains four structured directions in total. These known values help interpret the
example, but cross-validation is not required to recover them exactly in a finite noisy sample.

## Evaluate the search

The example declares `CHOSEN_N_COMPONENTS=2` before fitting the search. The search evaluates
admissible pairs of paired-mode count $h$ (`n_components`) and retained predictor-subspace dimension
$r_\pi$ (`predictor_rank`):

```python
--8<-- "examples/02_synthetic_path_selection.py:fit-synthetic-search"
```

For each paired-mode count, the component path retains the evaluated predictor rank with the
smallest mean response-standardized CV-MSE:

\begin{equation}
r_\pi^*(h)
=
\operatorname*{arg\,min}_{r_\pi}
\operatorname{CV\text{-}MSE}(h,r_\pi).
\end{equation}

At this stage the search owns validation evidence. It has not fitted a final model on all training
observations.

## Retrieve selection evidence

Retrieve the conditioned component path, create the manual selection at two components, and inspect
the predictor-rank profile for that selected component count:

```python
--8<-- "examples/02_synthetic_path_selection.py:inspect-synthetic-selection"
```

`selection` is the complete immutable row $[h,r_\pi^*(h)]$. These search-owned results can be
retrieved without fitting a final model. The selection is created before final fitting and will be
passed unchanged to `refit()`.

### Component path

```python
--8<-- "examples/02_synthetic_path_selection.py:plot-synthetic-component-path"
```

![Synthetic component path](../assets/generated/synthetic/component_path.svg)

The mean CV-MSE falls markedly from one to two components and changes little at three. The bars
show one population standard deviation across the materialized validation splits on either side
of each mean. They describe split-to-split variability; they are not confidence intervals and do
not enter selection. This tutorial keeps the component choice explicit. The diamond marks the
selected row that will be refitted.

### Conditional predictor-rank profile

```python
--8<-- "examples/02_synthetic_path_selection.py:plot-synthetic-rank-profile"
```

![Synthetic predictor-rank profile](../assets/generated/synthetic/predictor_rank_profile.svg)

At two components, the lowest evaluated mean CV-MSE occurs at predictor rank four. In this
controlled example, that matches the two shared and two predictor-specific directions in the
predictor block. This agreement is informative but not a general selection guarantee.

## Refit the selected pair

The final estimator consumes the already inspected selection:

```python
--8<-- "examples/02_synthetic_path_selection.py:refit-synthetic-model"
```

`refit(selection=selection)` does not resolve the component choice again. It verifies that the
selection belongs to the fitted search, fits its exact component-count and predictor-rank pair on
all training observations, and attaches the same immutable object as `model.selection_` after the
fit succeeds. The search remains the owner of the path evidence; the returned estimator owns
prediction and fitted-model inspection.

## Evaluate independent predictions

Predictions are calculated for the independent test observations after the selected pair has been
refitted on all training data:

```python
--8<-- "examples/02_synthetic_path_selection.py:evaluate-synthetic-predictions"
```

The completed diagnostic result is then rendered:

```python
--8<-- "examples/02_synthetic_path_selection.py:plot-synthetic-predictions"
```

![Synthetic observed versus predicted responses](../assets/generated/synthetic/observed_vs_predicted.svg)

The prediction plot uses the held-out test block, so it is not a fitted-data or
selection-conditioned OOF display. `model.score(test.X, test.Y)` supplies the corresponding uniform
average of the response-wise coefficients of determination.

## Minimal reusable workflow

For ordinary manual selection, the essential sequence is:

```python
from sklearn.model_selection import KFold

from pipls import PiPLSSearchCV

cv = KFold(n_splits=5, shuffle=True, random_state=0)
search = PiPLSSearchCV(cv=cv).fit(X_train, Y_train)

path = search.component_path_
selection = search.select(n_components=chosen_n_components)
rank_profile = search.predictor_rank_profile(selection.n_components)

model = search.refit(
    X_train,
    Y_train,
    selection=selection,
)
Y_pred = model.predict(X_test)
```

Use `model.selection_` to inspect the fitted estimator's retained provenance after refitting. Use
the pre-existing `selection` as the workflow handoff whenever search evidence, OOF reporting, and
final refitting must refer to exactly the same row.

## Continue with real data

The [complete Pulp tutorial](pulp.md) adds selection-conditioned OOF predictions, immutable
inspection objects, standard PLS-family plots, and Pi-PLS-specific factorization plots.

For exact signatures and advanced behavior, see:

- [`PiPLSSearchCV`](../api/path.md#pipls.PiPLSSearchCV);
- [`PiPLSRegression`](../api/regression.md#pipls.PiPLSRegression);
- [Path-selection details](../path_analysis.md).

## Reproduce this tutorial

The executable calculation is maintained in `examples/02_synthetic_path_selection.py`, and the
code blocks above are checked snippets from that file. From a source checkout, `make docs-figures`
regenerates the three SVG figures and `make docs` regenerates them before building the strict site.
See [Documentation reproducibility](../reproducibility.md#documentation-reproducibility) for the
required dependencies and distribution-level checks.
