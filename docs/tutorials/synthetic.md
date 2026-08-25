# Inspect and refit a manually selected Π-PLS model

This tutorial continues the workflow introduced in the [Pulp quick start](quick_start.md), but uses
deterministic synthetic training and test data so that the latent structure is known and prediction
assessment remains independent of model selection. It inspects
the validation evidence before choosing a component count, creates one explicit selection, and passes it to the final full-data refit.

For ordinary programming use, Π-PLS behaves like a one-parameter component search: as in PLS,
the main complexity parameter is `n_components`, denoted by $h$. A **component path** is the
sequence of cross-validated prediction errors obtained as $h$ is varied. The search resolves the
predictor rank $r_\pi$ internally for each $h$. Details can be reviewed under
[Interpretation of the ranks](../theory.md#interpretation-of-the-ranks).

The workflow is to generate independent training and test data, fit the search, inspect the
component path, choose a component count and create one selection, optionally inspect the
conditional predictor-rank profile, accept the selection, refit that same selection, and predict the
external test data. If the selected path or conditional rank evidence is unsatisfactory, return to
the selection step before accepting it.

```mermaid
flowchart TD
    data["Generate training and test data"]
    search["Fit search"]
    path["Inspect component path"]
    select["Choose component count and create selection"]
    review["Inspect selected path; optionally inspect rank profile"]
    refit["Refit the same selection"]
    predict["Predict external test data"]

    data --> search --> path --> select --> review --> refit --> predict
    review -. "revise" .-> select
```

The tutorial deliberately stops after one prediction plot. Scores, loadings, Π-PLS factorization
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

The details of the synthetic data generation is not important to learn Π-PLS regression analysis, but for the interested reader, the generating structure contains:

- two shared directions that affect both predictors and responses;
- two predictor-specific directions that affect only the predictors;
- one response-specific direction that affects only the responses.

The shared dimension, and therefore the intended predictive paired-mode count, is two, while the
predictor block contains four structured directions in total. These known values help interpret the
example, but cross-validation is not required to recover them exactly in a finite noisy sample.

## Fit the search

Fit the component search just as you would fit a PLS component search:

```python
--8<-- "examples/02_synthetic_path_selection.py:fit-synthetic-search"
```

The programming-level decision is how many paired latent modes to retain, so `n_components` is the
quantity displayed on the component path. Under the hood, Π-PLS also investigates predictor rank
$r_\pi$ conditionally for each $h$; that resolved rank is already incorporated into each
component-path row. Most users can therefore treat the search as a one-dimensional component-count
problem. We return to the predictor-rank search, with graphical support, under
[Optional: inspect the conditional predictor-rank profile](#inspect-conditional-predictor-rank-profile).

At this stage the search owns validation evidence. It has not selected a component count or fitted
a final model on all training observations.

## Inspect the component path { #retrieve-selection-evidence }

The component path is the PLS-like view of model complexity: one row for each evaluated $h$, with
the conditionally resolved predictor rank already incorporated. Retrieve it without creating a
selection:

```python
--8<-- "examples/02_synthetic_path_selection.py:inspect-synthetic-component-path"
```

Define a local component-path plotter once, then render the unconditional path:

```python
--8<-- "examples/02_synthetic_path_selection.py:define-synthetic-component-path-plotter"
```

```python
--8<-- "examples/02_synthetic_path_selection.py:plot-synthetic-component-path"
```

![Synthetic component path before selection](../assets/generated/synthetic/component_path.svg)

The mean CV-MSE falls markedly from one to two components and changes little at three. The bars
show one population standard deviation across the materialized validation splits on either side
of each mean. They describe split-to-split variability; they are not confidence intervals and do
not enter selection.

## Choose the component count and create the selection

Inspect the path and identify its elbow point. This which represents the simplest model that can be constructed significantly impairing predictive performance.   Record the number of components at that point and create the corresponding immutable search selection:

```python
--8<-- "examples/02_synthetic_path_selection.py:choose-synthetic-selection"
```

The static script records the resulting choice so that the complete example is reproducible. In an
interactive analysis, inspect the first path figure, set the value, and rerun from this selection
stage.

## Inspect the selected evidence

The selected component-path row can be reviewed without fitting a final model. For advanced
inspection, also retrieve the predictor-rank profile conditional on the chosen component count.
The same `path` object is reused for the selected presentation:

```python
--8<-- "examples/02_synthetic_path_selection.py:inspect-synthetic-selected-evidence"
```

`selection` is the complete immutable row $[h,r_\pi(h)]$, where $r_\pi(h)$ is the predictor rank
already resolved by the fitted search. The selection will be passed unchanged to `refit()` if the
evidence is accepted.

### Selected component path

```python
--8<-- "examples/02_synthetic_path_selection.py:plot-synthetic-selected-component-path"
```

![Synthetic selected component path](../assets/generated/synthetic/selected_component_path.svg)

The path is unchanged; the orange diamond identifies the selected two-component row without repeating the evaluation.

### Optional: inspect the conditional predictor-rank profile { #inspect-conditional-predictor-rank-profile }

Most users can make the model-complexity decision from the component path alone. This is where the
predictor-rank search mentioned earlier becomes visible. By default, `PiPLSSearchCV` evaluates every
fold-feasible integer predictor rank conditionally at the chosen $h$. Advanced users can inspect
those evaluations directly:

```python
--8<-- "examples/02_synthetic_path_selection.py:plot-synthetic-rank-profile"
```

![Synthetic predictor-rank profile](../assets/generated/synthetic/predictor_rank_profile.svg)

The profile shows the predictor ranks actually evaluated at the chosen $h$. With the default
scorer, larger configured scores are equivalent to smaller mean response-standardized CV-MSE.
`reference_selection` identifies the exact minimum-CV-MSE rank, while `selection` identifies the
smallest rank admitted by the fitted predictor-rank tolerance. With the default machine-scale
tolerance these are normally the same. Here both select predictor rank four, which
matches the two shared and two predictor-specific directions in the predictor block. This
agreement is informative but not a general selection guarantee.

Advanced analyses can control predictor rank through the search configuration. See
[Path and selection](../path_selection.md#predictor-rank-policies) for the available policies and tolerances.

The selected path and optional rank profile are the model-selection evidence in this workflow. If
they make the chosen component count unsatisfactory, revise `CHOSEN_N_COMPONENTS` and create a new
selection. Once accepted, keep that immutable selection fixed for final refitting and for any later
selection-conditioned OOF diagnosis.

## Refit the selected pair

The final estimator consumes the accepted selection:

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

The prediction plot uses the held-out test block. `model.score(test.X, test.Y)` supplies the corresponding uniform
average of the response-wise coefficients of determination.

## Minimal reusable workflow

For ordinary manual selection, the essential sequence is:

```python
from sklearn.model_selection import KFold

from pipls import PiPLSSearchCV

cv = KFold(n_splits=5, shuffle=True, random_state=0)
search = PiPLSSearchCV(cv=cv).fit(X_train, Y_train)

path = search.component_path_
# Inspect path before assigning chosen_n_components.
selection = search.select(n_components=chosen_n_components)
selected_path = search.component_path_
# Optionally, for advanced inspection:
rank_profile = search.predictor_rank_profile(selection.n_components)
# Inspect the selected evidence; revise selection if needed.

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
inspection objects, standard PLS-family plots, and Π-PLS-specific factorization plots.

For exact signatures and advanced behavior, see:

- [`PiPLSSearchCV`](../api/path.md#pipls.PiPLSSearchCV);
- [`PiPLSRegression`](../api/regression.md#pipls.PiPLSRegression);
- [Path and selection](../path_selection.md).

## Reproduce this tutorial

The executable calculation is maintained in `examples/02_synthetic_path_selection.py`, and the
code blocks above are checked snippets from that file. From a source checkout, `make docs-figures`
regenerates the four SVG figures and `make docs` regenerates them before building the strict site.
See [Documentation reproducibility](../reproducibility.md#documentation-reproducibility) for the
required dependencies and distribution-level checks.
