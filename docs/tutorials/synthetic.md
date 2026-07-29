# First Pi-PLS model with synthetic data

This tutorial shows the shortest complete path from data generation to a selected Pi-PLS model. It
uses deterministic synthetic training and test data so that the latent structure is known and the
prediction assessment is independent of model selection. For Pi-PLS, `n_components` is the number
of paired latent modes.

## What this tutorial covers

You will:

1. generate independent training and test observations;
2. evaluate a cross-validated component path;
3. choose a paired-mode count;
4. retrieve the predictor rank selected at that count;
5. fit one fixed `PiPLSRegression` model;
6. predict the independent test responses.

The tutorial deliberately stops after one prediction plot. Scores, loadings, Pi-PLS factorization
plots, and selection-conditioned OOF diagnostics are introduced in the
[complete Pulp tutorial](pulp.md).

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
example, but cross-validation
is not required to recover them exactly in a finite noisy sample.

## Evaluate the component path

`PiPLSSearchCV` evaluates admissible pairs of paired-mode count $h$ (`n_components`) and retained
predictor-subspace dimension $r_\pi$ (`predictor_rank`). For each paired-mode count, it selects the
evaluated predictor rank with the smallest mean
response-standardized CV-MSE under the default scorer:

\[
r_\pi^*(h)
=
\operatorname*{arg\,min}_{r_\pi}
\operatorname{CV\text{-}MSE}(h,r_\pi).
\]

The concise `component_path_` object contains one row per evaluated paired-mode count. Each row
already contains its conditionally selected predictor rank $r_\pi^*(h)$:

```python
--8<-- "examples/02_synthetic_path_selection.py:evaluate-synthetic-path"
```

This tutorial chooses `CHOSEN_N_COMPONENTS=2`. The call to `for_n_components()` only retrieves the
stored row; it does not perform another search and it does not fit the final model.

The component path is plotted before fitting:

```python
--8<-- "examples/02_synthetic_path_selection.py:plot-synthetic-component-path"
```

![Synthetic component path](../assets/generated/synthetic/component_path.svg)

The mean CV-MSE falls markedly from one to two components and changes little at three. The bars
show one fold-based standard error on either side of each mean; they are not confidence intervals.
Such bars can inform the conventional [one-standard-error rule](../path_analysis.md#one-standard-error-component-heuristic)
for choosing a parsimonious paired-mode count. This tutorial keeps that judgment explicit rather than
setting `selection_rule="one_standard_error"`. The diamond marks the choice of two components.

## Inspect the conditional predictor-rank profile

The selected paired-mode row stores one predictor rank, but the complete evaluated rank profile is
available through `predictor_rank_profile()`:

```python
--8<-- "examples/02_synthetic_path_selection.py:plot-synthetic-rank-profile"
```

![Synthetic predictor-rank profile](../assets/generated/synthetic/predictor_rank_profile.svg)

At two components, the lowest evaluated mean CV-MSE occurs at predictor rank four. In this
controlled example, that matches the two shared and two predictor-specific directions in the
predictor block. This agreement is informative but not a general selection guarantee.

The programming contract is now complete:

```text
search candidate pairs
        ↓
component_path_ stores one selected predictor rank for each paired-mode count
        ↓
choose a paired-mode count
        ↓
for_n_components(h) returns [h, r_pi*(h)]
        ↓
fit one fixed PiPLSRegression model
```

## Fit the selected model and predict

Only after inspecting the two selection figures is the fixed model fitted. Predictions are then
made for the independent test observations:

```python
--8<-- "examples/02_synthetic_path_selection.py:fit-predict-synthetic-model"
```

![Synthetic observed versus predicted responses](../assets/generated/synthetic/observed_vs_predicted.svg)

The prediction plot uses the held-out test block, so it is not a fitted-data or
selection-conditioned OOF display. `model.score(test.X, test.Y)` supplies the corresponding uniform average of the response-wise
coefficients of determination.

## Minimal reusable workflow

For ordinary use, the essential sequence is:

```python
search = PiPLSSearchCV().fit(X_train, Y_train)
selected = search.component_path_.for_n_components(chosen_n_components)

model = PiPLSRegression(
    n_components=selected.n_components,
    predictor_rank=selected.predictor_rank,
).fit(X_train, Y_train)

Y_pred = model.predict(X_test)
```

`PiPLSSearchCV` owns model selection. `PiPLSRegression` represents one fixed selected model. Keeping
those roles separate makes the workflow explicit and modifiable.

## Continue with real data

The [complete Pulp tutorial](pulp.md) applies the same selection sequence to a real multivariate
dataset. It then adds fixed-parameter OOF predictions, immutable inspection objects, standard
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
