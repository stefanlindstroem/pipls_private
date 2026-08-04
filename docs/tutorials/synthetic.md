# Inspect a manually selected Pi-PLS model with synthetic data

This tutorial expands the [Pulp quick start](quick_start.md) by retaining the fitted search object,
declaring a component count, fitting the corresponding model, and then inspecting the retained
component path and conditional predictor-rank profile. Deterministic synthetic training and test
data make the latent structure known and keep prediction assessment independent of model
selection. For Pi-PLS, `n_components` is the number of paired latent modes.

## What this tutorial covers

You will:

1. generate independent training and test observations;
2. declare a paired-mode count;
3. evaluate the search and fit the corresponding full-data model;
4. inspect the fitted model's selection, component path, and predictor-rank profile;
5. predict the independent test responses;
6. render the numerical results.

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

## Fit the manually selected model

The example declares `CHOSEN_N_COMPONENTS=2` before modeling. The search evaluates admissible pairs
of paired-mode count $h$ (`n_components`) and retained predictor-subspace dimension $r_\pi$
(`predictor_rank`). For each paired-mode count, it selects the evaluated predictor rank with the
smallest mean response-standardized CV-MSE:

\begin{equation}
r_\pi^*(h)
=
\operatorname*{arg\,min}_{r_\pi}
\operatorname{CV\text{-}MSE}(h,r_\pi).
\end{equation}

`refit()` resolves the stored row $[h,r_\pi^*(h)]$ for the declared component count and fits that
fixed pair on all training observations:

```python
--8<-- "examples/02_synthetic_path_selection.py:fit-synthetic-model"
```

At this point modeling is complete. The returned estimator owns prediction and fitted-model
inspection. The retained search owns the numerical evidence produced during path evaluation.

## Retrieve selection evidence

The exact row used by `refit()` is available as `model.selection_`. The component path and
conditional predictor-rank profile are then retrieved for analysis:

```python
--8<-- "examples/02_synthetic_path_selection.py:inspect-synthetic-selection"
```

No additional selection is performed. `selection`, `path`, and `rank_profile` are immutable
numerical results derived from the completed modeling workflow.

### Component path

```python
--8<-- "examples/02_synthetic_path_selection.py:plot-synthetic-component-path"
```

![Synthetic component path](../assets/generated/synthetic/component_path.svg)

The mean CV-MSE falls markedly from one to two components and changes little at three. The bars
show one fold-based standard error on either side of each mean; they are not confidence intervals.
Such bars can inform the conventional
[one-standard-error rule](../path_analysis.md#one-standard-error-component-heuristic). This tutorial
keeps the component choice explicit. The diamond marks the row that produced the fitted model.

### Conditional predictor-rank profile

```python
--8<-- "examples/02_synthetic_path_selection.py:plot-synthetic-rank-profile"
```

![Synthetic predictor-rank profile](../assets/generated/synthetic/predictor_rank_profile.svg)

At two components, the lowest evaluated mean CV-MSE occurs at predictor rank four. In this
controlled example, that matches the two shared and two predictor-specific directions in the
predictor block. This agreement is informative but not a general selection guarantee.

The programming contract is:

```text
search candidate pairs
        ↓
refit the declared component count using its stored conditional predictor rank
        ↓
model.selection_ records the exact fitted row
        ↓
component_path_ and predictor_rank_profile() expose retained selection evidence
```

## Evaluate independent predictions

Predictions are calculated for the independent test observations after the full-data model has been
constructed:

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

model = search.refit(
    X_train,
    Y_train,
    n_components=chosen_n_components,
)

selection = model.selection_
path = search.component_path_
rank_profile = search.predictor_rank_profile(selection.n_components)

Y_pred = model.predict(X_test)
```

After `refit()`, obtain the fitted selection from `model.selection_`. Use `search.select()` when a
selection is needed without fitting a final model.

## Continue with real data

The [complete Pulp tutorial](pulp.md) applies the same ordering to a real multivariate dataset. It
then adds OOF predictions for the fitted selection, immutable inspection objects, standard
PLS-family plots, and Pi-PLS-specific factorization plots.

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
