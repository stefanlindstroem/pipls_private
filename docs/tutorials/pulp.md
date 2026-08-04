# Pulp: a complete Pi-PLS workflow

This tutorial applies [Inspect a manually selected Pi-PLS model with synthetic data](synthetic.md)
to a real multivariate dataset. It assumes that `PiPLSSearchCV`, `component_path_`,
`predictor_rank_profile()`, and fixed-model fitting are already familiar. The focus is what changes
with real data: an interior predictor-rank result, selection-conditioned out-of-fold (OOF)
predictions, and interpretation of a selected model.

## What this tutorial covers

You will:

1. load the Pulp predictors and responses;
2. declare a component count and fit the corresponding model;
3. retrieve the model's selection, component path, and conditional predictor-rank profile;
4. generate OOF predictions for that exact selection;
5. compute immutable latent-structure, factorization, and prediction-diagnostic results;
6. render and interpret representative standard PLS-family and Pi-PLS-specific plots.

## Setup

Install the example dependencies before running the analysis from a source checkout:

```bash
python -m pip install ".[examples]"
```

The example imports the estimators, numerical inspection functions, Matplotlib, and `adjustText`,
then states the component choice and diagnostic-response limit used below:

```python
--8<-- "examples/05_pulp_real_data.py:pulp-tutorial-setup"
```

`CHOSEN_N_COMPONENTS=3` is the explicit modeling choice. Because every fitted component is shown,
the plotting code derives its zero-based component indices from that value instead of maintaining a
second display setting. The first three response columns are shown only in pointwise diagnostic
figures to keep the demonstration legible; the RMSE summary still includes all eight responses.

## The data and modeling question

The package-owned dataset contains 46 thermomechanical-pulp samples, 14 fiber-description
predictors, and eight responses comprising Canadian Standard Freeness and seven handsheet
properties. `load_pulp()` reads the installed resources without network access. The matrices are
adapted from supplementary material associated with Lindström et al. (2025); all rows are preserved,
and no imputation or learned preprocessing is applied before fitting. See the
[dataset description](../datasets.md#pulp-real-data-integration) and the [reference](#reference) for
provenance.

Predictor labels retain the source notation: `L` is contour length, `W` is width, `C` is the source
FiberLab C descriptor, and `F` is fibrillation. The suffixes `arith`, `lw`, and `llw` denote
arithmetic, length-weighted, and length-length-weighted means. The response labels are `CSF`
(Canadian Standard Freeness), `Density`, `TI` (tensile index), `Elongation`, `TEA` (tensile
energy absorption), `TSI` (tensile stiffness index), `Tear index`, and `s` (light-scattering
coefficient).

The analysis uses three paired latent modes and the predictor rank selected conditionally at that
component count. The distinction is summarized in
[Interpretation of the ranks](../theory.md#interpretation-of-the-ranks).

## Load the data

The named loader returns immutable matrices together with scientific predictor and response labels:

```python
--8<-- "examples/05_pulp_real_data.py:load-pulp-data"
```

The resulting arrays have shapes `(46, 14)` and `(46, 8)`. The same loader works from a source
checkout, wheel, or source distribution and applies no preprocessing.

## Fit the manually selected model

Search and full-data refitting are kept together:

```python
--8<-- "examples/05_pulp_real_data.py:fit-pulp-model"
```

The search evaluates admissible paired-mode counts and conditionally selects one predictor rank at
each count. `refit()` resolves the stored row for `CHOSEN_N_COMPONENTS` and fits that fixed pair on
all 46 observations. The returned [`PiPLSRegression`](../api/regression.md#pipls.PiPLSRegression)
supplies predictions and fitted-model inspection; the search retains the cross-validation evidence.

At this point modeling is complete.

## Retrieve selection evidence

The exact row used by `refit()` is available from `model.selection_`. The component path and the
complete predictor-rank profile at that selected component count are then retrieved for analysis:

```python
--8<-- "examples/05_pulp_real_data.py:inspect-pulp-selection"
```

After `refit()`, obtain the fitted selection from `model.selection_`. Use `search.select()` when a
selection is needed without fitting a final model.

### Component path

```python
--8<-- "examples/05_pulp_real_data.py:plot-pulp-component-path"
```

![Pulp component path](../assets/generated/pulp/component_path.svg)

The mean CV-MSE falls substantially through three components and is nearly flat thereafter. The
bars show one population standard deviation across the materialized validation splits on either
side of each mean. They describe split-to-split variability; they are not confidence intervals
and do not enter selection. This manual workflow keeps the three-component choice explicit. The
diamond marks the row used by the fitted model.

The selection contains `predictor_rank=9`, the rank with the lowest evaluated mean CV-MSE at three
components under the seeded shuffled folds.

### Conditional predictor-rank profile

```python
--8<-- "examples/05_pulp_real_data.py:plot-pulp-rank-profile"
```

![Pulp predictor-rank profile](../assets/generated/pulp/predictor_rank_profile.svg)

For these 46 rows, 14 predictors, and five-fold CV, the support rule gives
$r_{\pi,\mathrm{max}}=\min[14,35,\lceil46/5\rceil]=10$. The seeded shuffled folds select the
interior rank 9. Ranks 9 and 10 have mean CV-MSE values of approximately 0.288 and 0.302, with
population split SDs of approximately 0.078 and 0.060. Their mean difference is small relative
to the displayed split-to-split variability.

The profile supports rank 9 for this fitted model, but it does not establish a distinct scientific
advantage over nearby retained dimensions. The fixed model still contains three paired latent
modes; predictor rank 9 is the retained predictor-subspace dimension used to estimate those modes.
See [Path-selection details](../path_analysis.md) for other bounds and policies.

## Generate selection-conditioned OOF predictions

OOF reporting is optional post-model analysis. The report consumes the exact selection retained by
the model rather than resolving the component count again:

```python
--8<-- "examples/05_pulp_real_data.py:pulp-oof-predictions"
```

`oof_report()` reuses the exact five seeded shuffled splits materialized during path evaluation and
recomputes row-ordered predictions for `selection`. The fixed random seed makes that partition
reproducible while avoiding a fold assignment determined by row order. Replace the search splitter
with a grouped, temporal, or otherwise appropriate protocol when the sampling design carries
experimental structure.

!!! important "Validation scope"
    These are **selection-conditioned OOF predictions**. The selected rank pair is fitted on each
    stored training fold, but the same observations were already used to inspect the selection path.
    Nested cross-validation or an external test set is required for an independent estimate of
    post-selection performance. See
    [ordered out-of-fold predictions](../path_analysis.md#ordered-out-of-fold-predictions).

## Compute immutable inspection results

The fitted estimator and OOF predictions are converted to numerical result objects before any
figure is rendered:

```python
--8<-- "examples/05_pulp_real_data.py:pulp-inspection-results"
```

| Result | Question answered |
|---|---|
| `LatentStructure` | How are samples and variables represented by the fitted PLS-family model? |
| `PiPLSDisplayFactors` | What are the Pi-PLS-specific $\mathbf{P}$, $\mathbf{D}$, $\mathbf{Q}$, and $\mathbf{Q}\mathbf{D}$ factors? |
| `PredictionDiagnostics` | How do the selection-conditioned OOF predictions and residuals behave? |

The Pulp workflow uses `response_names.index("TI")` as the response sign anchor and requests a
positive orientation. The resulting TI entry is nonnegative for every displayed component, and the
same component sign is applied to the paired columns of $\mathbf{P}$ and $\mathbf{Q}$. This
convention is useful here because tensile index is the principal controlled target. It only chooses
how an equivalent factorization is displayed; it does not change predictions or assert that every
physical effect on TI is positive. If an anchored entry were exactly zero, the helper would use its
default predictor-based sign for that component.

At this point all numerical analysis is complete. The remaining code only renders completed public
result objects. The full catalogue is in [Model inspection](../model_inspection.md).

## Interpret representative fitted-model plots

Factor signs are arbitrary, so paired quantities may change sign together without changing
predictions. This tutorial fixes that ambiguity with the TI-positive convention above. Interpret
relative patterns and paired quantities rather than treating the chosen sign as a fitted scientific
conclusion.

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

#### Predictor directions $\mathbf{P}$

The grouped bars are constructed directly from `factors.predictor_directions`:

```python
--8<-- "tools/render_pulp_tutorial.py:render-pulp-predictor-directions"
```

![Pulp predictor directions](../assets/generated/pulp/predictor_directions.svg)

The dominant entries differ by component: the first direction emphasizes `Shives` and selected
fibrillation or length descriptors, the second emphasizes length descriptors, and the third is
strongly associated with `Fines B`. Only relative within-component patterns should be interpreted;
the displayed orientation is fixed by the TI entries in the paired response directions.

The columns of $\mathbf{P}$ are orthonormal predictor directions paired with orthonormal response
directions in $\mathbf{P}\mathbf{D}\mathbf{Q}^{\mathsf T}$. They are distinct from ordinary X
loadings. The figure shows all three selected paired latent modes; predictor rank 9 does not create
nine plotted modes. See
[Predictor directions](../model_inspection.md#predictor-directions) and
[Diagonal latent coupling](../theory.md#diagonal-latent-coupling).

#### Weighted response directions $\mathbf{Q}\mathbf{D}$

The grouped bars are constructed directly from `factors.weighted_response_directions`:

```python
--8<-- "tools/render_pulp_tutorial.py:render-pulp-weighted-response-directions"
```

![Pulp weighted response directions](../assets/generated/pulp/weighted_response_directions.svg)

The first component has its largest absolute entries for `TI`, `TEA`, `Tear index`, and `TSI`.
The second component is most pronounced for `Tear index` and `s`, while the third contrasts `CSF`
with `Elongation`. Because column $k$ of $\mathbf{Q}\mathbf{D}$ is $d_kQ_{:k}$, it combines each
response direction with the dilation of its paired latent mode and shows orientation and strength
rather than $\mathbf{Q}$ alone.

The complete example includes separate $\mathbf{D}$ and $\mathbf{Q}$ plots in the same
four-panel Pi-PLS factorization figure. See [Dilation](../model_inspection.md#dilation),
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

## Reproduce this tutorial

The analysis and selection snippets are maintained in `examples/05_pulp_real_data.py`. Run the
complete example from the repository root:

```bash
python examples/05_pulp_real_data.py
```

Standalone interpretation-figure recipes are maintained in `tools/render_pulp_tutorial.py`.
`make docs-figures` regenerates the eight representative single-chart SVGs displayed here, while
the numbered example writes six caller-owned PDFs with additional score, loading, factorization,
and coefficient views. Both routes calculate directly from in-memory results and write no
analytical CSV intermediates. See
[Documentation reproducibility](../reproducibility.md#documentation-reproducibility) for the
strict documentation-build and source-distribution checks.

## Next steps

- Use [Model inspection](../model_inspection.md) for the complete quantity catalogue and
  interpretation boundaries.
- Use [Path-selection details](../path_analysis.md) for nondefault bounds, policies, pipelines,
  scorer behavior, grouped or temporal splitters, leave-one-out interpretation, OOF coverage,
  and automatic refitting.
- Use [Examples](../examples.md) for Sugarcane, Tobacco, and the ordinary-PLS path comparison.
- Use the [`PiPLSSearchCV` reference](../api/path.md#pipls.PiPLSSearchCV) and
  [inspection API](../api/inspection.md) for exact signatures.

## Reference

Stefan B. Lindström, Rita Ferritsius, Johan E. Carlson, Johan Persson, and Fritjof Nilsson,
“Predicting handsheet properties and enhancing refiner control using fiber analyzer data and
latent
variable modeling,” *Computers & Chemical Engineering* **199** (2025), 109143,
[doi:10.1016/j.compchemeng.2025.109143](https://doi.org/10.1016/j.compchemeng.2025.109143).
