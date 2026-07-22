# Pulp: a complete Pi-PLS workflow

This tutorial develops one Pi-PLS model from data loading through parameter selection, fixed fitting,
out-of-fold prediction, and fitted-model interpretation. It uses the Pulp dataset because its 14
named predictors and eight responses are large enough to show the multivariate workflow while the
plots remain readable.

The repository generates every figure on this page from the same executable workflow used by
`examples/10_pulp_real_data.py`. Run `make docs-figures` to regenerate the SVG files, or `make docs`
to regenerate them and build the complete site.

## The modeling problem

The data contain 46 thermomechanical-pulp samples:

- 14 fiber-description predictors in `datasets/pulp/X.csv`;
- Canadian Standard Freeness and seven handsheet-property responses in `datasets/pulp/Y.csv`.

The tables are adapted from supplementary material associated with Lindström et al. (2025). The
repository preserves all 46 rows and applies no imputation or learned preprocessing before fitting.
See the [dataset description](../datasets.md#pulp-real-data-integration) and the
[reference](#reference) for provenance.

Pi-PLS represents the predictive relation through paired predictor and response latent variables.
Two parameters control the fitted model:

- `n_components` is the number of paired latent variables retained for prediction;
- `predictor_rank` is the dimension of the predictor subspace available when those pairs are
  estimated.

The two values are selected together. The workflow first evaluates candidate component counts. At
each count, `PiPLSPathCV` computes response-standardized MSE for every admissible predictor rank in
each validation fold and selects the rank that minimizes the mean across folds. This mean is the
reported CV-MSE. The final model is then fitted with one explicit rank pair. See the
[theory overview](../theory.md#interpretation-of-the-ranks) for the mathematical distinction.

## Load the data

The numbered example reads the two committed CSV files directly with pandas:

```python
--8<-- "examples/10_pulp_real_data.py:load-pulp-data"
```

The resulting data frames have shapes `(46, 14)` and `(46, 8)`. Their headers become the scientific
labels used in the figures. No package-specific dataset container or hidden workflow object is
required.

## Evaluate the component path

The path search is fitted directly to the two response and predictor tables:

```python
--8<-- "examples/10_pulp_real_data.py:evaluate-pulp-component-path"
```

The default `n_components_values="all"` evaluates every admissible component count. For each count,
the search compares the admissible predictor ranks by mean response-standardized CV-MSE. The path
row records the rank attaining the lowest evaluated mean, that minimum CV-MSE, and the fold-to-fold
standard deviation at the minimizing rank.

The purpose of the component-path figure is to choose the number of components. This tutorial uses
`CHOSEN_N_COMPONENTS=3` as an explicit elbow-based choice. Before drawing the figure, the matching
immutable row is retrieved so that the chosen point can be marked:

```python
--8<-- "examples/10_pulp_real_data.py:select-pulp-parameters"
```

`for_n_components()` does not fit a model or repeat the minimization. It returns the row already
evaluated for the stated component count. Consequently, `selected.predictor_rank` is the predictor
rank that minimized mean CV-MSE conditional on `selected.n_components`. The same scalar result is
used later when the fixed model is fitted.

The path can then be plotted with ordinary Matplotlib:

```python
--8<-- "examples/10_pulp_real_data.py:plot-pulp-component-path"
```

![Pulp component path](../assets/generated/pulp/component_path.svg)

The curve falls substantially through three components and then changes little. The marker shows
the tutorial's choice of `n_components=3`. This is a stated modeling choice, not an automatic rule
that three components are always optimal. The absolute minimum, fold variation, parsimony, and
scientific purpose should all be considered.

The selected row records `predictor_rank=10` because rank 10 gives the lowest evaluated mean CV-MSE
for three components. That conditional minimum is examined next and is used only when the fixed
model is fitted. The predictor rank must be taken from the same row as the chosen component count.
See [Advanced path-search behavior](../path_analysis.md) for nondefault search policies and the
[`PiPLSPathCV` reference](../api/path.md#pipls.PiPLSPathCV) for the exact result contract.

## Inspect the conditional predictor-rank profile

The selected row summarizes a second calculation: for three components, the path evaluates
predictor ranks 3 through 10 and chooses the rank with the lowest mean CV-MSE. The complete
one-dimensional profile is retrieved without filtering `cv_results_` manually:

```python
--8<-- "examples/10_pulp_real_data.py:extract-pulp-rank-profile"
```

The returned arrays are sorted by predictor rank and are read-only. `rank_profile.selected` is the
same conditional scalar result already stored in `selected`. Under a non-default scorer, the
selected row maximizes the configured mean score; the CV-MSE arrays remain descriptive diagnostics.

The conditional profile is plotted directly from those immutable arrays:

```python
--8<-- "examples/10_pulp_real_data.py:plot-pulp-rank-profile"
```

![Pulp predictor-rank profile](../assets/generated/pulp/predictor_rank_profile.svg)

Rank 10 has the lowest mean loss within the evaluated range, but it is also the upper search
boundary. Ranks 9 and 10 have mean CV-MSE values of approximately 0.347 and 0.331, while their fold
standard deviations are approximately 0.167 and 0.151. The difference between the two means is
therefore small relative to the fold variation. Rank 10 is the path-selected value for the fixed
model, but this profile does not establish that ranks above 10 would be worse or that rank 10 has a
scientifically distinct advantage over rank 9.

The fitted model still has three paired Pi-PLS components. The predictor rank of 10 describes the
dimension of the predictor basis available when those three pairs are estimated; it does not mean
that the factor plots contain ten paired components. See
[Interpretation of the ranks](../theory.md#interpretation-of-the-ranks).

## Fit the selected model

Only after the component-path and predictor-rank figures have been inspected is a new fixed
estimator fitted to all 46 samples. The previously retrieved scalar result supplies both members of
the fixed rank pair:

```python
--8<-- "examples/10_pulp_real_data.py:fit-pulp-model"
```

The path-search object and the fitted estimator have different roles: the first supports model
selection, while the second represents the chosen model and supplies predictions and inspection
results. `PiPLSRegression` learns predictor and response centering and scaling inside every fit; no
external scaler is used here. The
[`PiPLSRegression` reference](../api/regression.md#pipls.PiPLSRegression) documents preprocessing,
solver, fitted-state, and method contracts.

## Generate out-of-fold predictions

Fitted values are unsuitable for judging predictive residuals. The example therefore uses the
standard scikit-learn mechanism to clone the fixed estimator inside five non-shuffled folds and
predict each held-out observation once:

```python
--8<-- "examples/10_pulp_real_data.py:pulp-oof-predictions"
```

These are labeled **selection-conditioned OOF predictions**. The rank pair is fixed during this
second cross-validation calculation, but the same 46 observations were previously used to inspect
the component path. The resulting diagnostics describe the selected model on these data; they are
not an independent estimate of post-selection performance. Use nested cross-validation or an
external test set when that stronger claim is required. See
[Cross-validation](../cross_validation.md) for the validation boundary.

## Compute inspection results

The fitted estimator and OOF predictions are converted to immutable inspection results before any
plotting takes place:

```python
--8<-- "examples/10_pulp_real_data.py:pulp-inspection-results"
```

`LatentStructure` contains the shared PLS-family scores, loadings, and regression coefficients.
`PiPLSDisplayFactors` contains $P$, $D$, $Q$, and $QD$ from the Pi-PLS factorization.
`PredictionDiagnostics` contains standardized prediction and residual quantities together with their
provenance. Plotting functions consume these results but do not modify the fitted model.

## Interpret the fitted model

The figures below are descriptive views of one selected full-data fit or of its
selection-conditioned OOF predictions. Singular-vector signs are arbitrary, so a component and its
associated directions may change sign together without changing predictions. Interpretation should
focus on relative patterns, paired quantities, predictions, and reconstruction rather than on an
isolated sign.

### Standard PLS-family latent-structure plots

These plots use the estimator-neutral `LatentStructure` result and apply to compatible fitted
PLS-family models, including both Pi-PLS and ordinary PLS.

#### Scores

![Pulp X scores](../assets/generated/pulp/scores.svg)

The score plot places each sample in the coordinate system defined by the first two fitted predictor
latent variables. Nearby points have similar coordinates in this two-component representation;
separation indicates differences represented by these components. The visible structure is
exploratory. The plot does not infer sample groups, confidence regions, or outliers.

See [Scores](../model_inspection.md#scores),
[`latent_structure()`](../api/inspection.md#pipls.inspection.latent_structure), and
[`plot_scores()`](../api/plotting.md#pipls.plotting.plot_scores).

#### Score-loading biplot

![Pulp score-loading biplot](../assets/generated/pulp/biplot.svg)

The biplot combines balanced sample-score coordinates with arrows derived from the X loadings for
the same two components. Predictors pointing in similar displayed directions have similar loading
patterns in this plane. A sample lying in the direction of an arrow has a positive coordinate along
that displayed predictor direction. These are geometric statements about the fitted two-component
view, not causal claims or automatic measures of variable importance.

See [Score-loading biplot](../model_inspection.md#score-loading-biplot),
[`biplot_coordinates()`](../api/inspection.md#pipls.inspection.biplot_coordinates), and
[`plot_biplot()`](../api/plotting.md#pipls.plotting.plot_biplot).

#### X loadings

![Pulp X loadings](../assets/generated/pulp/x_loadings.svg)

X loadings describe how the original predictor variables reconstruct the fitted predictor scores.
The three components emphasize different combinations of fiber variables. Large positive or
negative bars indicate strong participation in that component's reconstruction, but the sign can
flip with the component basis. Loadings are not regression coefficients and do not by themselves
measure predictive importance.

See [X loadings](../model_inspection.md#x-loadings)
and [`plot_x_loadings()`](../api/plotting.md#pipls.plotting.plot_x_loadings).

#### Y loadings

![Pulp Y loadings](../assets/generated/pulp/y_loadings.svg)

Y loadings show how each response participates in the response-side latent representation. Responses
with similar loading patterns across components are represented similarly in this fitted latent
space. Because the responses were scaled during fitting, these bars describe the standardized model
rather than response values in their original units.

See [Y loadings](../model_inspection.md#y-loadings)
and [`plot_y_loadings()`](../api/plotting.md#pipls.plotting.plot_y_loadings).

### Pi-PLS-specific factorization plots

These plots use `PiPLSDisplayFactors` and expose quantities specific to the Pi-PLS factorization.
They do not have direct counterparts in ordinary PLS.

#### Predictor directions $P$

![Pulp predictor directions](../assets/generated/pulp/predictor_directions.svg)

The columns of $P$ are Pi-PLS predictor rotations. They define the predictor-side directions paired
with the response directions. Their pattern differs from the ordinary X loadings because $P$ belongs
to the regression factorization $PDQ^{\mathsf T}$ rather than to score reconstruction.

See [Predictor directions](../model_inspection.md#predictor-directions),
[Diagonal latent coupling](../theory.md#diagonal-latent-coupling), and
[`plot_pipls_predictor_directions()`](../api/plotting.md#pipls.plotting.plot_pipls_predictor_directions).

#### Dilation $D$

![Pulp dilation](../assets/generated/pulp/dilation.svg)

The diagonal values $d_k=D_{kk}$ scale the paired predictor and response directions. In this fit,
the first value is clearly larger than the second and third, so the first paired mode carries the
largest scaling in the centered and standardized regression map. Dilation is only one part of a
mode; its scientific effect must be read together with the corresponding columns of $P$ and $Q$.

See [Dilation](../model_inspection.md#dilation) and
[`plot_pipls_dilation()`](../api/plotting.md#pipls.plotting.plot_pipls_dilation).

#### Response directions $Q$

![Pulp response directions](../assets/generated/pulp/response_directions.svg)

The columns of $Q$ are response-side rotations paired with the columns of $P$. They show the relative
response pattern of each mode before dilation. A direction may have a pronounced response pattern
even when its corresponding dilation is modest.

See [Response directions](../model_inspection.md#response-directions) and
[`plot_pipls_response_directions()`](../api/plotting.md#pipls.plotting.plot_pipls_response_directions).

#### Weighted response directions $QD$

![Pulp weighted response directions](../assets/generated/pulp/weighted_response_directions.svg)

$QD$ multiplies each response direction by its dilation. This view therefore combines response-side
orientation and mode strength. It is useful when comparing how the fitted modes enter the regression
map, while $Q$ remains the clearer view of direction alone.

See [Weighted response directions](../model_inspection.md#weighted-response-directions) and
[`plot_pipls_weighted_response_directions()`](../api/plotting.md#pipls.plotting.plot_pipls_weighted_response_directions).

### Standard PLS-family prediction plots

These plots use `PredictionDiagnostics` and are standard prediction checks for a fitted PLS-family
model. Here they retain the explicit selection-conditioned OOF provenance described above.

#### Observed versus predicted

![Pulp observed versus predicted](../assets/generated/pulp/observed_vs_predicted.svg)

Observed and OOF-predicted values are standardized response by response so that `CSF`, `Density`,
and `TI` can share one axis. Agreement is read relative to the identity line. The spread around that
line describes selection-conditioned OOF error for the displayed responses; it does not turn these
predictions into an independent test set.

See [Observed versus predicted](../model_inspection.md#observed-versus-predicted) and
[`plot_observed_vs_predicted()`](../api/plotting.md#pipls.plotting.plot_observed_vs_predicted).

#### Residuals versus predicted

![Pulp residuals versus predicted](../assets/generated/pulp/residuals_vs_predicted.svg)

This plot places standardized residuals against standardized OOF predictions. The zero line marks
perfect agreement. Curvature, changing spread, or response-specific bands can indicate structure
not captured by the fitted linear model. Individual points should not be labeled as anomalous from
this plot alone.

See [Residuals versus predicted](../model_inspection.md#residuals-versus-predicted) and
[`plot_residuals_vs_predicted()`](../api/plotting.md#pipls.plotting.plot_residuals_vs_predicted).

#### Standardized RMSE

![Pulp standardized RMSE](../assets/generated/pulp/standardized_rmse.svg)

The response-wise RMSE values use the complete observed-response sample standard deviations as
display scales. This permits a direct descriptive comparison across all eight responses: lower bars
indicate smaller OOF error relative to the observed spread of that response. These values are not
identical to the fold-local standardized losses used during component-path selection.

See [Standardized RMSE](../model_inspection.md#standardized-rmse) and
[`plot_standardized_rmse()`](../api/plotting.md#pipls.plotting.plot_standardized_rmse).

## Common variations

The workflow above keeps the model-selection sequence explicit. The same public estimators also
cover several common variations without introducing a second package-specific workflow.

### Fit one known rank pair directly

When both ranks are already fixed by an external protocol, fit `PiPLSRegression` directly:

```python
from pipls import PiPLSRegression

model = PiPLSRegression(
    n_components=2,
    predictor_rank=4,
).fit(X_train, Y_train)

Y_pred = model.predict(X_test)
```

This estimator performs no internal parameter selection. It always centers predictors and
responses and, with `scale=True`, learns sample standard deviations from the current training data.
A failed fit leaves the estimator unfitted rather than preserving partial or earlier fitted state.

### Change the validation design or scorer

`cv=` accepts ordinary scikit-learn splitters, and group-aware splitters receive `groups` through
`fit()`:

```python
from sklearn.model_selection import GroupKFold

search = PiPLSPathCV(
    cv=GroupKFold(n_splits=5),
    scoring="neg_mean_absolute_error",
    refit=False,
).fit(X, Y, groups=sample_groups)
```

Selection always maximizes the configured mean test score. The CV-MSE-minimization interpretation
used throughout this tutorial is specific to the default negative response-standardized-MSE scorer.
See [Cross-validation](../cross_validation.md) for OOF coverage, repeated folds, temporal splits,
and leave-one-out interpretation.

### Add learned preprocessing

Pi-PLS centering and optional scaling already occur inside every estimator fit. Additional learned
preprocessing should be placed before `PiPLSRegression` in a supported scikit-learn pipeline so that
the complete pipeline is cloned and fitted separately inside every validation fold. Do not fit such
preprocessing on the complete dataset before path evaluation. See
[Advanced path-search behavior](../path_analysis.md#pipelines-and-fold-local-preprocessing).

### Refit the globally best evaluated pair

Set `refit=True` when the scorer-defined global best evaluated candidate should be fitted
automatically on all supplied data:

```python
search = PiPLSPathCV(refit=True).fit(X, Y)
Y_pred = search.predict(X_new)
```

The tutorial uses `refit=False` because it chooses the component count from the path explicitly and
then fits one fixed model. With `refit=True`, the selected pair is determined entirely by the
configured scorer and tie-breaking rules.

## Complete executable example

The maintained example contains the complete calculation and figure composition directly:

```python
--8<-- "examples/10_pulp_real_data.py"
```

Run it from the repository root with:

```bash
python examples/10_pulp_real_data.py
```

The generated tutorial SVGs and the example PDFs serve different purposes. The tutorial uses one
chart per SVG for explanation and omits the raw regression-coefficient chart because the predictors
and responses use heterogeneous original units that make a single tutorial figure difficult to
read. The numbered example still demonstrates that plotting API and groups related charts into six
caller-owned PDF figures: `component_path.pdf`, `predictor_rank_profile.pdf`, `pipls_factors.pdf`,
`latent_structure.pdf`, `coefficients.pdf`, and `prediction_diagnostics.pdf`. Both routes calculate
from in-memory results and write no generated analytical CSV files.

## Next steps

- Use [Advanced path-search behavior](../path_analysis.md) for nondefault bounds, policies,
  pipelines, refitting, and detailed result surfaces.
- Use [Model inspection](../model_inspection.md) for general definitions that are not specific to
  Pulp.
- Use [Examples](../examples.md) for Sugarcane, Tobacco, synthetic data, and the ordinary-PLS path
  comparison.
- Use [Theory](../theory.md) for the implemented matrix construction and rank constraints.

## Reference

Stefan B. Lindström, Rita Ferritsius, Johan E. Carlson, Johan Persson, and Fritjof Nilsson,
“Predicting handsheet properties and enhancing refiner control using fiber analyzer data and latent
variable modeling,” *Computers & Chemical Engineering* **199** (2025), 109143,
[doi:10.1016/j.compchemeng.2025.109143](https://doi.org/10.1016/j.compchemeng.2025.109143).
