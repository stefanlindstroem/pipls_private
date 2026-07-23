# Pulp: a complete real-data analysis

This tutorial applies the workflow from [First Pi-PLS model with synthetic data](synthetic.md) to a
real multivariate dataset. It assumes that the roles of `PiPLSPathCV`, `component_path_`,
`predictor_rank_profile()`, and the fixed `PiPLSRegression` estimator are already familiar. The focus
here is what changes with real data: an upper-boundary predictor-rank choice, selection-conditioned
out-of-fold predictions, and scientific interpretation of a selected model.

Every code snippet comes from `examples/10_pulp_real_data.py`. Run `make docs-figures` to regenerate
the six SVG figures on this page, or `make docs` to regenerate them and build the complete site.

## The data and modeling question

The dataset contains 46 thermomechanical-pulp samples:

- 14 fiber-description predictors in `datasets/pulp/X.csv`;
- Canadian Standard Freeness and seven handsheet-property responses in `datasets/pulp/Y.csv`.

The tables are adapted from supplementary material associated with Lindström et al. (2025). The
repository preserves all rows and applies no imputation or learned preprocessing before fitting.
See the [dataset description](../datasets.md#pulp-real-data-integration) and the
[reference](#reference) for provenance.

The analysis asks for a parsimonious number of paired Pi-PLS components, then uses the predictor
rank selected conditionally at that component count. The mathematical distinction between these
ranks is summarized in [Interpretation of the ranks](../theory.md#interpretation-of-the-ranks).

## Load the data

The example reads the committed predictor and response tables directly with pandas:

```python
--8<-- "examples/10_pulp_real_data.py:load-pulp-data"
```

The resulting data frames have shapes `(46, 14)` and `(46, 8)`. Their headers supply the scientific
labels used in the plots. No package-specific loader or workflow object is required.

## Select the fixed rank pair

The synthetic tutorial introduced the selection contract. The same two plots are retained here
because the real-data choice requires additional qualification.

### Component path

The search evaluates the admissible component counts and their conditional predictor-rank minima:

```python
--8<-- "examples/10_pulp_real_data.py:evaluate-pulp-component-path"
```

This tutorial uses `CHOSEN_N_COMPONENTS=3` as an explicit elbow-based choice. The matching immutable
row is retrieved before plotting so that the selected point can be marked:

```python
--8<-- "examples/10_pulp_real_data.py:select-pulp-parameters"
```

```python
--8<-- "examples/10_pulp_real_data.py:plot-pulp-component-path"
```

![Pulp component path](../assets/generated/pulp/component_path.svg)

The mean CV-MSE falls substantially through three components and then changes little. The diamond
marks the stated choice of three components; it is not an automatic rule. Fold standard deviation
is descriptive variability rather than a confidence interval.

The selected row contains `predictor_rank=10`, the rank with the lowest evaluated mean CV-MSE at
three components. `for_n_components()` only retrieves that evaluated row; it does not repeat the
optimization or fit the final model.

### Conditional predictor-rank profile

The complete rank profile at three components is available without filtering `cv_results_`:

```python
--8<-- "examples/10_pulp_real_data.py:extract-pulp-rank-profile"
```

```python
--8<-- "examples/10_pulp_real_data.py:plot-pulp-rank-profile"
```

![Pulp predictor-rank profile](../assets/generated/pulp/predictor_rank_profile.svg)

Rank 10 has the lowest evaluated mean CV-MSE, but it is also the upper search boundary. Ranks 9 and
10 have mean CV-MSE values of approximately 0.347 and 0.331, with fold standard deviations of
approximately 0.167 and 0.151. Their mean difference is small relative to the fold variation. The
profile supports using rank 10 for this fitted model, but it does not establish that ranks above 10
would be worse or that rank 10 has a distinct scientific advantage over rank 9.

The fixed model still contains three paired components. Predictor rank 10 is the dimension of the
predictor basis used to estimate those pairs; it is not the number of displayed components. See
[Advanced path-search behavior](../path_analysis.md) for alternative search policies.

## Fit the selected model

Only after the two selection plots have been inspected is a new fixed estimator fitted to all 46
samples:

```python
--8<-- "examples/10_pulp_real_data.py:fit-pulp-model"
```

The search object supports model selection; the fixed estimator supplies predictions and fitted
results. `PiPLSRegression` learns predictor and response centering and scaling inside the fit. The
[`PiPLSRegression` reference](../api/regression.md#pipls.PiPLSRegression) gives the exact estimator
contract.

## Generate selection-conditioned OOF predictions

Fitted values are unsuitable for assessing predictive residuals. The example therefore clones the
fixed estimator inside five non-shuffled folds and predicts each held-out observation once:

```python
--8<-- "examples/10_pulp_real_data.py:pulp-oof-predictions"
```

These are **selection-conditioned OOF predictions**. The rank pair is fixed during this second
cross-validation calculation, but the same 46 observations were already used to inspect the
selection path. The diagnostics describe the selected model on these data; they are not an
independent estimate of post-selection performance. Nested cross-validation or an external test set
is required for that stronger claim. See [Cross-validation](../cross_validation.md).

## Compute immutable inspection results

The fitted estimator and OOF predictions are converted to numerical result objects before plotting:

```python
--8<-- "examples/10_pulp_real_data.py:pulp-inspection-results"
```

| Result | Question answered |
|---|---|
| `LatentStructure` | How are samples and variables represented by the fitted PLS-family model? |
| `PiPLSDisplayFactors` | What are the Pi-PLS-specific $P$, $D$, $Q$, and $QD$ factors? |
| `PredictionDiagnostics` | How do the selection-conditioned OOF predictions and residuals behave? |

At this point the programming workflow is complete: the rank pair has been selected, the fixed model
has been fitted, OOF predictions have been calculated, and reusable numerical results are available.
The remaining figures are optional interpretation views. The full catalogue is in
[Model inspection](../model_inspection.md).

## Interpret representative fitted-model plots

The plots below are representative rather than exhaustive. Singular-vector signs are arbitrary, so
paired quantities may change sign together without changing predictions. Interpret relative
patterns and paired quantities, not isolated signs.

### Standard PLS-family latent-structure plot

#### Score-loading biplot

The balancing calculation remains package-owned, but the chart is ordinary Matplotlib. Predictor
labels are standard text artists, and `adjustText` moves them after the axis has been fully
configured:

```python
--8<-- "examples/10_pulp_real_data.py:plot-pulp-biplot"
```

![Pulp score-loading biplot](../assets/generated/pulp/biplot.svg)

The biplot combines balanced sample-score coordinates with predictor arrows from the first two X
loading columns. Nearby samples have similar displayed latent coordinates, while predictors pointing
in similar directions have similar loading patterns in this plane. Arrow length and angle are
specific to the displayed scaling and should not be read as regression coefficients or formal
variable importance.

See [Score-loading biplot](../model_inspection.md#score-loading-biplot) and
[`biplot_coordinates()`](../api/inspection.md#pipls.inspection.biplot_coordinates).

### Pi-PLS-specific factorization plot

#### Predictor directions $P$

![Pulp predictor directions](../assets/generated/pulp/predictor_directions.svg)

The columns of $P$ are Pi-PLS predictor rotations paired with response directions in the regression
factorization $PDQ^{\mathsf T}$. They are distinct from ordinary X loadings, which describe score
reconstruction. The figure shows all three selected paired components; predictor rank 10 does not
create ten plotted components.

See [Predictor directions](../model_inspection.md#predictor-directions) and
[Diagonal latent coupling](../theory.md#diagonal-latent-coupling). The numbered example plots
$P$, $D$, $Q$, and $QD$ directly from `PiPLSDisplayFactors`; the model-inspection guide explains
the complementary views.

### Standard PLS-family prediction plots

These figures use the arrays stored in `PredictionDiagnostics` directly. The example owns the
Matplotlib series, reference lines, labels, and response selection:

```python
--8<-- "examples/10_pulp_real_data.py:plot-pulp-prediction-diagnostics"
```

The relevant fields are `observed_standardized`, `predicted_standardized`,
`residual_standardized`, `standardized_rmse`, and `prediction_kind`.

#### Observed versus predicted

![Pulp observed versus predicted](../assets/generated/pulp/observed_vs_predicted.svg)

Observed and OOF-predicted values are standardized response by response so that `CSF`, `Density`,
and `TI` can share one axis. Agreement is read relative to the identity line. The spread describes
selection-conditioned OOF error for the displayed responses, not independent-test performance.

See [Observed versus predicted](../model_inspection.md#observed-versus-predicted).

#### Standardized RMSE

![Pulp standardized RMSE](../assets/generated/pulp/standardized_rmse.svg)

Response-wise RMSE is divided by the observed sample standard deviation of each response, allowing a
descriptive comparison across all eight responses. These values are not identical to the fold-local
standardized losses used during component-path selection.

See [Standardized RMSE](../model_inspection.md#standardized-rmse).

## Complete executable example

The maintained example contains the full calculation and a broader plotting demonstration:

```python
--8<-- "examples/10_pulp_real_data.py"
```

Run it from the repository root with:

```bash
python examples/10_pulp_real_data.py
```

The tutorial renderer writes six representative single-chart SVGs. The numbered example remains a
complete API demonstration and writes six caller-owned PDFs, including additional score, loading,
factorization, residual, and coefficient views. Both routes calculate directly from in-memory
results and write no generated analytical CSV files.

## Next steps

- Use [Model inspection](../model_inspection.md) for the complete plot catalogue and interpretation
  boundaries.
- Use [Advanced path-search behavior](../path_analysis.md) for nondefault bounds, policies,
  pipelines, scorer behavior, and automatic refitting.
- Use [Cross-validation](../cross_validation.md) for grouped, repeated, temporal, leave-one-out, and
  OOF-coverage details.
- Use [Examples](../examples.md) for Sugarcane, Tobacco, and the ordinary-PLS path comparison.
- Use the [`PiPLSPathCV` reference](../api/path.md#pipls.PiPLSPathCV) and
  [inspection API](../api/inspection.md) for exact numerical-result signatures.

## Reference

Stefan B. Lindström, Rita Ferritsius, Johan E. Carlson, Johan Persson, and Fritjof Nilsson,
“Predicting handsheet properties and enhancing refiner control using fiber analyzer data and latent
variable modeling,” *Computers & Chemical Engineering* **199** (2025), 109143,
[doi:10.1016/j.compchemeng.2025.109143](https://doi.org/10.1016/j.compchemeng.2025.109143).
