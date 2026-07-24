# Pulp: a complete Pi-PLS workflow

This tutorial applies [First Pi-PLS model with synthetic data](synthetic.md) to a real
multivariate dataset. It assumes that `PiPLSPathCV`, `component_path_`,
`predictor_rank_profile()`, and fixed-model fitting are already familiar. The focus is what changes
with real data: an upper-boundary predictor-rank result, selection-conditioned out-of-fold (OOF)
predictions, and interpretation of a selected model.

Analysis and selection snippets come from `examples/05_pulp_real_data.py`. Standalone
interpretation-figure recipes come from `tools/render_pulp_tutorial.py`, which generates the eight
SVG figures displayed here. Install the
example dependencies before running either route:

```bash
python -m pip install -e ".[examples]"
```

## What this tutorial covers

You will:

1. load the Pulp predictors and responses;
2. evaluate the component path and conditional predictor-rank profile;
3. qualify a predictor-rank choice at the upper evaluated boundary;
4. fit one fixed `PiPLSRegression` model;
5. generate selection-conditioned OOF predictions;
6. compute immutable latent-structure, factorization, and prediction-diagnostic results;
7. interpret representative standard PLS-family and Pi-PLS-specific plots.

## Setup

The example imports the estimators, numerical inspection functions, Matplotlib, and `adjustText`,
then states the component and display choices used below:

```python
--8<-- "examples/05_pulp_real_data.py:pulp-tutorial-setup"
```

`CHOSEN_N_COMPONENTS=3` is the user choice made from the component path.
`DISPLAY_COMPONENTS=(0, 1, 2)` uses Python's zero-based indices for the three fitted components.
The first three response columns are shown in pointwise diagnostic figures only to keep the
demonstration legible; the RMSE summary still includes all eight responses.

## The data and modeling question

The dataset contains 46 thermomechanical-pulp samples:

- 14 fiber-description predictors in `datasets/pulp/X.csv`;
- Canadian Standard Freeness and seven handsheet-property responses in `datasets/pulp/Y.csv`.

The tables are adapted from supplementary material associated with Lindström et al. (2025). The
repository preserves all rows and applies no imputation or learned preprocessing before fitting.
See the [dataset description](../datasets.md#pulp-real-data-integration) and the
[reference](#reference) for provenance.

Predictor labels retain the source notation: `L` is contour length, `W` is width, `C` is the source
FiberLab C descriptor, and `F` is fibrillation. The suffixes `arith`, `lw`, and `llw` denote
arithmetic, length-weighted, and length-length-weighted means. The response labels are `CSF`
(Canadian Standard Freeness), `Density`, `TI` (tensile index), `Elongation`, `TEA` (tensile
energy absorption), `TSI` (tensile stiffness index), `Tear index`, and `s` (light-scattering
coefficient).

The analysis asks for a parsimonious number of paired Pi-PLS components, then uses the predictor
rank selected conditionally at that component count. The distinction is summarized in
[Interpretation of the ranks](../theory.md#interpretation-of-the-ranks).

## Load the data

The committed tables are read directly with pandas. Their headers supply the scientific labels, and
the first three response indices are selected explicitly for the pointwise displays:

```python
--8<-- "examples/05_pulp_real_data.py:load-pulp-data"
```

The resulting data frames have shapes `(46, 14)` and `(46, 8)`. No package-specific loader or
workflow object is required.

## Select the fixed rank pair

### Component path

The search evaluates admissible component counts and selects one predictor rank conditionally at
each count:

```python
--8<-- "examples/05_pulp_real_data.py:evaluate-pulp-component-path"
```

The matching immutable row is retrieved before plotting so that the chosen point can be marked:

```python
--8<-- "examples/05_pulp_real_data.py:select-pulp-parameters"
```

```python
--8<-- "examples/05_pulp_real_data.py:plot-pulp-component-path"
```

![Pulp component path](../assets/generated/pulp/component_path.svg)

The mean CV-MSE falls substantially through three components and is nearly flat thereafter. The
diamond marks the stated elbow-based choice; it is not an automatic rule. Fold standard deviation
is descriptive variability rather than a confidence interval.

The selected row contains `predictor_rank=10`, the rank with the lowest evaluated mean CV-MSE at
three components. `for_n_components()` retrieves that evaluated row; it does not repeat the
optimization or fit the final model.

### Conditional predictor-rank profile

The complete evaluated rank profile at three components is available without filtering
`cv_results_`:

```python
--8<-- "examples/05_pulp_real_data.py:extract-pulp-rank-profile"
```

```python
--8<-- "examples/05_pulp_real_data.py:plot-pulp-rank-profile"
```

![Pulp predictor-rank profile](../assets/generated/pulp/predictor_rank_profile.svg)

Rank 10 has the lowest evaluated mean CV-MSE, but it is also the upper default boundary. For these
46 rows, 14 predictors, and five-fold CV, the support rule gives
$r_{\pi,\max}=\min[14,35,\lceil46/5\rceil]=10$. Ranks 9 and 10 have mean CV-MSE values of
approximately 0.347 and 0.331, with fold standard deviations of approximately 0.167 and 0.151.
Their mean difference is small relative to the fold variation.

The profile supports rank 10 for this fitted model, but it does not establish that ranks above 10
would be worse or that rank 10 has a distinct scientific advantage over rank 9. The fixed model
still contains three paired components; predictor rank 10 is the dimension of the predictor basis
used to estimate those pairs. See [Path-selection details](../path_analysis.md) for other
bounds and policies.

## Fit the selected model

Only after the two selection plots have been inspected is a fixed estimator fitted to all 46
samples:

```python
--8<-- "examples/05_pulp_real_data.py:fit-pulp-model"
```

The search object supports model selection; the fixed estimator supplies predictions and fitted
results. `PiPLSRegression` learns predictor and response centering and scaling inside the fit. The
[`PiPLSRegression` reference](../api/regression.md#pipls.PiPLSRegression) gives the exact contract.

## Generate selection-conditioned OOF predictions

Fitted values are unsuitable for assessing predictive residuals. The example therefore clones the
fixed estimator inside five non-shuffled folds and predicts each held-out observation once:

```python
--8<-- "examples/05_pulp_real_data.py:pulp-oof-predictions"
```

The non-shuffled splitter is used here for a deterministic demonstration. Replace it with a grouped,
temporal, or otherwise appropriate splitter when row order carries experimental structure.

!!! important "Validation scope"
    These are **selection-conditioned OOF predictions**. The rank pair is fixed during this second
    cross-validation calculation, but the same observations were already used to inspect the
    selection path. Nested cross-validation or an external test set is required for an independent
    estimate of post-selection performance. See
    [ordered out-of-fold predictions](../path_analysis.md#ordered-out-of-fold-predictions).

## Compute immutable inspection results

The fitted estimator and OOF predictions are converted to numerical result objects before plotting:

```python
--8<-- "examples/05_pulp_real_data.py:pulp-inspection-results"
```

| Result | Question answered |
|---|---|
| `LatentStructure` | How are samples and variables represented by the fitted PLS-family model? |
| `PiPLSDisplayFactors` | What are the Pi-PLS-specific $P$, $D$, $Q$, and $QD$ factors? |
| `PredictionDiagnostics` | How do the selection-conditioned OOF predictions and residuals behave? |

At this point the programming workflow is complete. The remaining figures are optional
interpretation views; the full catalogue is in [Model inspection](../model_inspection.md).

## Interpret representative fitted-model plots

Singular-vector signs are arbitrary, so paired quantities may change sign together without changing
predictions. Interpret relative patterns and paired quantities, not isolated signs.

### Standard PLS-family latent structure

#### Score-loading biplot

`biplot_coordinates()` supplies balanced numerical coordinates. Matplotlib draws samples and
predictor arrows, and [`adjustText`](https://adjusttext.readthedocs.io/) moves the labels after
the axis has been configured:

```python
--8<-- "tools/render_pulp_tutorial.py:render-pulp-biplot"
```

![Pulp score-loading biplot](../assets/generated/pulp/biplot.svg)

The three length descriptors point in closely similar directions in the displayed plane, while
`Shives` contrasts with several C descriptors. These are loading-pattern relationships under the
chosen biplot scaling, not regression coefficients or formal variable importance.

See [Score-loading biplot](../model_inspection.md#score-loading-biplot) and
[`biplot_coordinates()`](../api/inspection.md#pipls.inspection.biplot_coordinates).

### Pi-PLS-specific factorization

#### Predictor directions $P$

The grouped bars are constructed directly from `factors.predictor_directions`:

```python
--8<-- "tools/render_pulp_tutorial.py:render-pulp-predictor-directions"
```

![Pulp predictor directions](../assets/generated/pulp/predictor_directions.svg)

The dominant entries differ by component: the first direction emphasizes `Shives` and selected
fibrillation or length descriptors, the second emphasizes length descriptors, and the third is
strongly associated with `Fines B`. Only relative within-component patterns should be interpreted;
signs may reverse together.

The columns of $P$ are predictor rotations paired with response directions in $PDQ^{\mathsf T}$.
They are distinct from ordinary X loadings. The figure shows all three selected paired components;
predictor rank 10 does not create ten plotted components. See
[Predictor directions](../model_inspection.md#predictor-directions) and
[Diagonal latent coupling](../theory.md#diagonal-latent-coupling).

#### Weighted response directions $QD$

The grouped bars are constructed directly from `factors.weighted_response_directions`:

```python
--8<-- "tools/render_pulp_tutorial.py:render-pulp-weighted-response-directions"
```

![Pulp weighted response directions](../assets/generated/pulp/weighted_response_directions.svg)

The first component has its largest absolute entries for `TI`, `TEA`, `Tear index`, and `TSI`.
The second component is most pronounced for `Tear index` and `s`, while the third contrasts `CSF`
with `Elongation`. Because $QD$ combines each response direction with its dilation, it shows the
response-side orientation and strength of the paired modes rather than $Q$ alone.

The complete example includes separate $D$ and $Q$ plots in the same four-panel Pi-PLS
factorization figure. See [Dilation](../model_inspection.md#dilation),
[Response directions](../model_inspection.md#response-directions), and
[Weighted response directions](../model_inspection.md#weighted-response-directions).

### Standard PLS-family prediction diagnostics

The pointwise figures show the first three response columns (`CSF`, `Density`, and `TI`) solely for
visibility. The summary retains all responses. All charts use named arrays from
`PredictionDiagnostics` directly.

#### Observed versus predicted

```python
--8<-- "tools/render_pulp_tutorial.py:render-pulp-observed-vs-predicted"
```

![Pulp observed versus predicted](../assets/generated/pulp/observed_vs_predicted.svg)

`CSF` lies more tightly around the identity line than `Density` and `TI`; all three figures remain
selection-conditioned rather than independent-test results.

See [Observed versus predicted](../model_inspection.md#observed-versus-predicted).

#### Residual versus predicted

```python
--8<-- "tools/render_pulp_tutorial.py:render-pulp-residuals-vs-predicted"
```

![Pulp residual versus predicted](../assets/generated/pulp/residuals_vs_predicted.svg)

No dominant global curvature is apparent in the displayed responses, although `TI` has the largest
residual excursions. The zero line is descriptive; it does not establish a formal variance model or
calibration claim.

See [Residuals versus predicted](../model_inspection.md#residuals-versus-predicted).

#### Standardized RMSE

```python
--8<-- "tools/render_pulp_tutorial.py:render-pulp-standardized-rmse"
```

![Pulp standardized RMSE](../assets/generated/pulp/standardized_rmse.svg)

Response-wise RMSE is divided by the observed sample standard deviation. `CSF` has the lowest value
(approximately 0.30), while `Tear index` has the highest (approximately 0.68). These values are not
identical to the fold-local standardized losses used during path selection.

See [Standardized RMSE](../model_inspection.md#standardized-rmse).

## Run the complete example

The maintained source is `examples/05_pulp_real_data.py`. Run it from the repository root:

```bash
python examples/05_pulp_real_data.py
```

The tutorial renderer writes eight representative single-chart SVGs. The numbered example writes
six caller-owned PDFs, including additional score, loading, factorization, and coefficient views.
Both routes calculate directly from in-memory results and write no analytical CSV intermediates.

## Next steps

- Use [Model inspection](../model_inspection.md) for the complete quantity catalogue and
  interpretation boundaries.
- Use [Path-selection details](../path_analysis.md) for nondefault bounds, policies, pipelines,
  scorer behavior, grouped or temporal splitters, leave-one-out interpretation, OOF coverage,
  and automatic refitting.
- Use [Examples](../examples.md) for Sugarcane, Tobacco, and the ordinary-PLS path comparison.
- Use the [`PiPLSPathCV` reference](../api/path.md#pipls.PiPLSPathCV) and
  [inspection API](../api/inspection.md) for exact signatures.

## Reference

Stefan B. Lindström, Rita Ferritsius, Johan E. Carlson, Johan Persson, and Fritjof Nilsson,
“Predicting handsheet properties and enhancing refiner control using fiber analyzer data and
latent
variable modeling,” *Computers & Chemical Engineering* **199** (2025), 109143,
[doi:10.1016/j.compchemeng.2025.109143](https://doi.org/10.1016/j.compchemeng.2025.109143).
