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

The two values are selected together. The workflow first evaluates candidate component counts. For
each count, `PiPLSPathCV` selects a predictor rank and reports response-standardized cross-validated
mean squared error (CV-MSE). The final model is then fitted with one explicit rank pair. See the
[theory overview](../theory.md#interpretation-of-the-ranks) for the mathematical distinction.

## Load the data

The canonical workflow reads the two committed CSV files directly with pandas:

```python
--8<-- "examples/_support/pulp_workflow.py:load-pulp-data"
```

The resulting data frames have shapes `(46, 14)` and `(46, 8)`. Their headers are retained as the
scientific labels used in tables and figures. No package-specific dataset container or hidden loader
is required.

## Construct the pipeline

The model is placed at the end of a scikit-learn `Pipeline`:

```python
--8<-- "examples/_support/pulp_workflow.py:build-pulp-pipeline"
```

This is intentionally a one-step pipeline. `PiPLSRegression` learns predictor and response
centering and scaling inside every fit. Adding an external predictor scaler here would either repeat
predictor preprocessing or require changing the estimator's response-scaling behavior. Additional
application-specific transformers may be inserted before the terminal `pipls` step when they are
scientifically justified.

The initial rank values only make the pipeline cloneable. They are replaced after the component
path has been examined. The search evaluates the complete pipeline, so any future preprocessing
steps would also be fitted separately inside each training fold. See
[Preprocessing](../preprocessing.md) and the
[`PiPLSRegression` reference](../api/regression.md#pipls.PiPLSRegression).

## Evaluate the component path

The complete pipeline is supplied to `PiPLSPathCV` with `refit=False`:

```python
--8<-- "examples/_support/pulp_workflow.py:evaluate-pulp-component-path"
```

The default `n_components_values="all"` evaluates every admissible component count. For each row,
the search records the conditionally selected predictor rank, mean response-standardized CV-MSE,
and fold-to-fold standard deviation.

![Pulp component path](../assets/generated/pulp/component_path.svg)

The curve falls substantially through three components and then changes little. The tutorial
therefore uses the visible elbow at `n_components=3`. The selected predictor rank on that row is
`predictor_rank=10`.

This is a stated modeling choice, not an automatic rule that three components are always optimal.
The absolute minimum, the fold variation, parsimony, and the scientific purpose of the model should
all be considered. The predictor rank must be taken from the same row as the chosen component count.
See [Parameter selection](../parameter_selection.md) and the
[`PiPLSPathCV` reference](../api/path.md#pipls.PiPLSPathCV).

## Choose and fit the fixed model

The predictor rank is read from the selected path row:

```python
--8<-- "examples/_support/pulp_workflow.py:select-pulp-predictor-rank"
```

The pipeline template is then cloned, assigned the fixed rank pair through nested pipeline
parameters, and fitted to all 46 samples:

```python
--8<-- "examples/_support/pulp_workflow.py:fit-pulp-pipeline"
```

The fitted pipeline is the object used for prediction. Its terminal fitted estimator is available as
`fitted_pipeline.named_steps["pipls"]`. The path-search object and the fixed fitted pipeline have
different roles: the first supports model selection, while the second represents the chosen model.

## Generate out-of-fold predictions

Fitted values are unsuitable for judging predictive residuals. The tutorial therefore clones the
fixed pipeline inside five non-shuffled folds and predicts each held-out observation once:

```python
--8<-- "examples/_support/pulp_workflow.py:pulp-oof-predictions"
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
--8<-- "examples/_support/pulp_workflow.py:pulp-inspection-results"
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

### Scores

![Pulp X scores](../assets/generated/pulp/scores.svg)

The score plot places each sample in the coordinate system defined by the first two fitted predictor
latent variables. Nearby points have similar coordinates in this two-component representation;
separation indicates differences represented by these components. The visible structure is
exploratory. The plot does not infer sample groups, confidence regions, or outliers.

See [Shared PLS-family latent structure](../model_inspection.md#shared-pls-family-latent-structure),
[`latent_structure()`](../api/inspection.md#pipls.inspection.latent_structure), and
[`plot_scores()`](../api/plotting.md#pipls.plotting.plot_scores).

### Score-loading biplot

![Pulp score-loading biplot](../assets/generated/pulp/biplot.svg)

The biplot combines balanced sample-score coordinates with arrows derived from the X loadings for
the same two components. Predictors pointing in similar displayed directions have similar loading
patterns in this plane. A sample lying in the direction of an arrow has a positive coordinate along
that displayed predictor direction. These are geometric statements about the fitted two-component
view, not causal claims or automatic measures of variable importance.

See [Shared PLS-family score-loading biplot](../model_inspection.md#shared-pls-family-score-loading-biplot),
[`biplot_coordinates()`](../api/inspection.md#pipls.inspection.biplot_coordinates), and
[`plot_biplot()`](../api/plotting.md#pipls.plotting.plot_biplot).

### X loadings

![Pulp X loadings](../assets/generated/pulp/x_loadings.svg)

X loadings describe how the original predictor variables reconstruct the fitted predictor scores.
The three components emphasize different combinations of fiber variables. Large positive or
negative bars indicate strong participation in that component's reconstruction, but the sign can
flip with the component basis. Loadings are not regression coefficients and do not by themselves
measure predictive importance.

See [Shared PLS-family latent structure](../model_inspection.md#shared-pls-family-latent-structure)
and [`plot_x_loadings()`](../api/plotting.md#pipls.plotting.plot_x_loadings).

### Y loadings

![Pulp Y loadings](../assets/generated/pulp/y_loadings.svg)

Y loadings show how each response participates in the response-side latent representation. Responses
with similar loading patterns across components are represented similarly in this fitted latent
space. Because the responses were scaled during fitting, these bars describe the standardized model
rather than response values in their original units.

See [Shared PLS-family latent structure](../model_inspection.md#shared-pls-family-latent-structure)
and [`plot_y_loadings()`](../api/plotting.md#pipls.plotting.plot_y_loadings).

### Predictor directions $P$

![Pulp predictor directions](../assets/generated/pulp/predictor_directions.svg)

The columns of $P$ are Pi-PLS predictor rotations. They define the predictor-side directions paired
with the response directions. Their pattern differs from the ordinary X loadings because $P$ belongs
to the regression factorization $PDQ^{\mathsf T}$ rather than to score reconstruction.

See [Pi-PLS display factors](../model_inspection.md#pi-pls-display-factors),
[Diagonal latent coupling](../theory.md#diagonal-latent-coupling), and
[`plot_pipls_predictor_directions()`](../api/plotting.md#pipls.plotting.plot_pipls_predictor_directions).

### Dilation $D$

![Pulp dilation](../assets/generated/pulp/dilation.svg)

The diagonal values $d_k=D_{kk}$ scale the paired predictor and response directions. In this fit,
the first value is clearly larger than the second and third, so the first paired mode carries the
largest scaling in the centered and standardized regression map. Dilation is only one part of a
mode; its scientific effect must be read together with the corresponding columns of $P$ and $Q$.

See [Pi-PLS display factors](../model_inspection.md#pi-pls-display-factors) and
[`plot_pipls_dilation()`](../api/plotting.md#pipls.plotting.plot_pipls_dilation).

### Response directions $Q$

![Pulp response directions](../assets/generated/pulp/response_directions.svg)

The columns of $Q$ are response-side rotations paired with the columns of $P$. They show the relative
response pattern of each mode before dilation. A direction may have a pronounced response pattern
even when its corresponding dilation is modest.

See [Pi-PLS display factors](../model_inspection.md#pi-pls-display-factors) and
[`plot_pipls_response_directions()`](../api/plotting.md#pipls.plotting.plot_pipls_response_directions).

### Weighted response directions $QD$

![Pulp weighted response directions](../assets/generated/pulp/weighted_response_directions.svg)

$QD$ multiplies each response direction by its dilation. This view therefore combines response-side
orientation and mode strength. It is useful when comparing how the fitted modes enter the regression
map, while $Q$ remains the clearer view of direction alone.

See [Pi-PLS display factors](../model_inspection.md#pi-pls-display-factors) and
[`plot_pipls_weighted_response_directions()`](../api/plotting.md#pipls.plotting.plot_pipls_weighted_response_directions).

### Regression coefficients

![Pulp regression coefficients](../assets/generated/pulp/coefficients.svg)

The coefficient plot shows the fitted linear map in the original predictor and response units for
`CSF`, `Density`, and `TI`. A coefficient gives the modeled response change associated with one
predictor-unit change while the other predictors are held fixed in the linear model. Because the
predictors and responses use different physical units, raw coefficient magnitudes should not be
compared across variables as a universal importance ranking. Correlated predictors can also share
or exchange coefficient weight.

The response subset is only a display choice; all eight responses were fitted. See
[Shared PLS-family latent structure](../model_inspection.md#shared-pls-family-latent-structure) and
[`plot_coefficients()`](../api/plotting.md#pipls.plotting.plot_coefficients).

### Observed versus predicted

![Pulp observed versus predicted](../assets/generated/pulp/observed_vs_predicted.svg)

Observed and OOF-predicted values are standardized response by response so that `CSF`, `Density`,
and `TI` can share one axis. Agreement is read relative to the identity line. The spread around that
line describes selection-conditioned OOF error for the displayed responses; it does not turn these
predictions into an independent test set.

See [Prediction diagnostics](../model_inspection.md#prediction-diagnostics) and
[`plot_observed_vs_predicted()`](../api/plotting.md#pipls.plotting.plot_observed_vs_predicted).

### Residuals versus predicted

![Pulp residuals versus predicted](../assets/generated/pulp/residuals_vs_predicted.svg)

This plot places standardized residuals against standardized OOF predictions. The zero line marks
perfect agreement. Curvature, changing spread, or response-specific bands can indicate structure
not captured by the fitted linear model. Individual points should not be labeled as anomalous from
this plot alone.

See [Prediction diagnostics](../model_inspection.md#prediction-diagnostics) and
[`plot_residuals_vs_predicted()`](../api/plotting.md#pipls.plotting.plot_residuals_vs_predicted).

### Standardized RMSE

![Pulp standardized RMSE](../assets/generated/pulp/standardized_rmse.svg)

The response-wise RMSE values use the complete observed-response sample standard deviations as
display scales. This permits a direct descriptive comparison across all eight responses: lower bars
indicate smaller OOF error relative to the observed spread of that response. These values are not
identical to the fold-local standardized losses used during component-path selection.

See [Prediction diagnostics](../model_inspection.md#prediction-diagnostics) and
[`plot_standardized_rmse()`](../api/plotting.md#pipls.plotting.plot_standardized_rmse).

## Complete executable example

The maintained example calls the same canonical workflow and then writes the component path,
post-analysis tables, and a multipage PDF report:

```python
--8<-- "examples/10_pulp_real_data.py"
```

Run it from the repository root with:

```bash
python examples/10_pulp_real_data.py
```

The generated tutorial SVGs and the example's PDF report serve different purposes. The tutorial uses
one chart per SVG for explanation. The numbered example demonstrates caller-owned panel composition
and canonical analysis artifacts.

## Next steps

- Use [Parameter selection](../parameter_selection.md) for all path-search options and tie-breaking
  rules.
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
