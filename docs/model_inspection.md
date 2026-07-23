# Model inspection

`pipls.inspection` computes immutable numerical results from fitted models or explicit predictions.
These arrays are the primary inspection interface. Maintained examples render them with ordinary
Matplotlib so the reader can see which quantities are displayed and can control every graphical
choice directly.

Pi-PLS deliberately provides no plotting submodule. The numerical objects can be rendered with
Matplotlib, another graphics system, or not rendered at all. `adjustText` is an optional external
label-layout aid for annotated biplots; it is not part of the numerical result contract.

Pi-PLS-specific inspection covers $P$, $D$, $Q$, and $QD$. Scores, loadings, coefficients, biplots,
observation diagnostics, and prediction diagnostics use estimator-neutral PLS-family objects.
The [Pulp tutorial](tutorials/pulp.md#interpret-representative-fitted-model-plots) uses a small
representative subset in one real-data analysis.

## Inspection results

### Pi-PLS display factors

A fitted `PiPLSRegression` stores the centered and scaled regression map as

\begin{equation}
B_{\mathrm{cs}}=PDQ^{\mathsf T}.
\end{equation}

`pipls_display_factors()` returns defensive read-only copies and applies one deterministic display
sign per component. The same sign is applied to the paired columns of $P$ and $Q$, so the regression
map is unchanged. The result contains:

- `predictor_directions`: $P$;
- `dilation`: the diagonal of $D$;
- `response_directions`: $Q$;
- `weighted_response_directions`: $QD$.

```python
from pipls.inspection import pipls_display_factors

factors = pipls_display_factors(model.decomposition_)
```

$P$ contains predictor rotations, not X loadings. X loadings belong to score reconstruction and are
stored separately in `LatentStructure`.

### Latent structure

`latent_structure()` copies public fitted PLS-family arrays into a read-only `LatentStructure`:

- `x_scores`, shape `(n_samples, n_components)`;
- `x_loadings`, shape `(n_features, n_components)`;
- `y_loadings`, shape `(n_targets, n_components)`;
- `coefficients`, shape `(n_targets, n_features)`.

```python
from pipls.inspection import latent_structure

structure = latent_structure(model)
```

The helper does not recompute, rescale, or sign-adjust these quantities. They describe the supplied
full-data fit and are not validation results.

### Biplot coordinates

`biplot_coordinates()` balances two selected score and X-loading columns while preserving their
rank-two reconstruction. For component $k$ it uses

\begin{equation}
a_k=\sqrt{\frac{\lVert p_k\rVert_2}{\lVert t_k\rVert_2}},\qquad
\widetilde t_k=a_kt_k,\qquad
\widetilde p_k=\frac{p_k}{a_k}.
\end{equation}

The result contains sample coordinates and predictor-arrow coordinates. It does not infer groups,
confidence regions, or variable importance.

### Prediction diagnostics

`prediction_diagnostics()` receives observed and predicted responses explicitly. It stores original
and response-standardized observations, predictions, residuals, response centers and scales,
response-wise standardized RMSE, and a required provenance label.

```python
from pipls.inspection import prediction_diagnostics

diagnostics = prediction_diagnostics(
    Y_test,
    model.predict(X_test),
    prediction_kind="external test predictions",
)
```

Standardization uses the supplied observed responses and is only for comparable displays. It does
not change predictions in their original units. Selection-conditioned OOF predictions are
descriptive post-selection diagnostics, not an independent estimate of future performance.

### Observation diagnostics

`observation_diagnostics()` transforms explicit predictor observations with a fitted PLS-family
model and returns:

- squared score distance from the training-score center, using the Moore--Penrose inverse of the
  training-score covariance;
- squared X-reconstruction residual from the model's public transform/inverse-transform round trip.

These are raw descriptive quantities. The package does not attach theoretical limits, automatic
outlier labels, or contribution diagnostics.

## Data-first plotting recipes

The table summarizes the numerical field and an ordinary Matplotlib primitive. The primitive is a
suggestion rather than a prescribed chart.

| Question | Numerical data | Typical Matplotlib operation |
|---|---|---|
| Where are samples in the latent plane? | `structure.x_scores` | `Axes.scatter()` |
| How do predictors reconstruct scores? | `structure.x_loadings` | `Axes.plot()` or `Axes.bar()` |
| How do responses enter the latent representation? | `structure.y_loadings` | `Axes.bar()` |
| What is the original-unit linear map? | `structure.coefficients` | `Axes.plot()` or `Axes.bar()` |
| Which observations are distant or poorly reconstructed? | two `ObservationDiagnostics` arrays | `Axes.scatter()` |
| How well do predictions agree with observations? | `PredictionDiagnostics` arrays | `Axes.scatter()` and `Axes.bar()` |

### Scores { #scores }

Select two columns of `structure.x_scores` and pass them directly to `Axes.scatter()`:

```python
first, second = 0, 1
axis.scatter(
    structure.x_scores[:, first],
    structure.x_scores[:, second],
    alpha=0.75,
)
axis.axhline(0.0, linewidth=0.8, linestyle="--")
axis.axvline(0.0, linewidth=0.8, linestyle="--")
axis.set_xlabel(f"X score component {first + 1}")
axis.set_ylabel(f"X score component {second + 1}")
```

Proximity means similar coordinates in that displayed latent plane. The chart is exploratory and
does not establish groups, confidence regions, or outliers.

### Score-loading biplot { #score-loading-biplot }

`biplot_coordinates()` calculates balanced sample and predictor coordinates for two components.
Plot `sample_coordinates` with `Axes.scatter()`, draw vectors from the origin to
`predictor_coordinates`, and create predictor labels with `Axes.text()`. For dense labels,
`adjustText.adjust_text()` can move those text artists after titles, limits, aspect, and legends have
been configured. Automatic placement is heuristic, so dense diagrams may still need manual work.

Similar arrow directions indicate similar loading patterns in the displayed plane. These are
geometric statements, not causal effects or automatic importance measures.

API: [`biplot_coordinates()`](api/inspection.md#pipls.inspection.biplot_coordinates).

### X loadings { #x-loadings }

Use selected columns of `structure.x_loadings`. A physical predictor coordinate can be passed to
`Axes.plot()` for spectral data; named predictors can be grouped with `Axes.bar()`.

```python
for component in (0, 1):
    axis.plot(
        wavelengths,
        structure.x_loadings[:, component],
        label=f"Component {component + 1}",
    )
```

Large absolute values indicate strong participation in score reconstruction. X loadings are not
regression coefficients and do not by themselves measure predictive importance. Component signs
may reverse without changing the model.

### Y loadings { #y-loadings }

Use selected columns of `structure.y_loadings`, normally with response names on a categorical axis.
Responses with similar patterns are represented similarly across the selected components. When the
model scales responses, the loadings describe the standardized fitted representation rather than
response values in original units.

### Predictor directions $P$ { #predictor-directions }

Use columns of `factors.predictor_directions` directly. Named predictors can be grouped with
`Axes.bar()`; spectral predictors can be plotted against their physical coordinate with
`Axes.plot()`:

```python
for component in components:
    axis.plot(
        predictor_coordinate,
        factors.predictor_directions[:, component],
        label=f"Component {component + 1}",
    )
```

The columns of $P$ define predictor rotations paired with response rotations in
$PDQ^{\mathsf T}$. They are distinct from X loadings because they belong to the regression
factorization rather than score reconstruction.

Theory: [Diagonal latent coupling](theory.md#diagonal-latent-coupling).

### Dilation $D$ { #dilation }

Plot `factors.dilation` with `Axes.bar()`, using component numbers on the categorical axis. Each
value $d_k=D_{kk}$ scales one paired predictor-response mode and should be interpreted together with
the matching columns of $P$ and $Q$.

```python
positions = np.arange(factors.n_components)
axis.bar(positions, factors.dilation)
axis.set_xticks(positions)
axis.set_xticklabels([f"Component {index + 1}" for index in positions])
```

### Response directions $Q$ { #response-directions }

Use columns of `factors.response_directions` with explicit response labels. Grouped bars make the
component comparison visible without hiding the chosen widths or offsets:

```python
for series, component in enumerate(components):
    offset = (series - (len(components) - 1) / 2.0) * width
    axis.bar(
        response_positions + offset,
        factors.response_directions[:, component],
        width=width,
        label=f"Component {component + 1}",
    )
```

The columns of $Q$ describe the response rotation of each paired mode before dilation.

### Weighted response directions $QD$ { #weighted-response-directions }

Plot `factors.weighted_response_directions` with the same response positions, widths, and component
selection used for $Q$. The array combines response-side orientation and mode strength through
$d_kq_{:k}$.

### Regression coefficients { #regression-coefficients }

Use rows of `structure.coefficients` directly. Each row describes one response in original predictor
and response units.

```python
for response, name in enumerate(response_names):
    axis.plot(
        predictor_coordinate,
        structure.coefficients[response],
        label=name,
    )
```

Raw magnitudes are not directly comparable across variables with different units, and correlated
predictors may share coefficient weight.

### Observed versus predicted { #observed-versus-predicted }

Use `PredictionDiagnostics.observed_standardized` and
`PredictionDiagnostics.predicted_standardized` with `Axes.scatter()`. Plot an identity line with
`Axes.plot()` after calculating limits from the displayed responses. Keep `prediction_kind` visible
in the title, caption, or surrounding report.

### Residuals versus predicted { #residuals-versus-predicted }

Use `PredictionDiagnostics.predicted_standardized` on the horizontal axis and
`PredictionDiagnostics.residual_standardized` on the vertical axis. Add the zero-residual reference
with `Axes.axhline()`.

### Standardized RMSE { #standardized-rmse }

Use `PredictionDiagnostics.standardized_rmse` with `Axes.bar()` or another caller-chosen summary.
Lower values mean smaller error relative to that response's observed spread. These values are not
generally equal to the mean fold-local standardized loss used during path selection.

### Observation diagnostics { #observation-diagnostics }

Plot `ObservationDiagnostics.score_distance` against
`ObservationDiagnostics.x_reconstruction_residual` with `Axes.scatter()`:

```python
axis.scatter(
    observations.score_distance,
    observations.x_reconstruction_residual,
    alpha=0.75,
)
```

Large values identify observations that are distant in the fitted score space, poorly reconstructed
in predictor space, or both. The package supplies no automatic thresholds or labels.

## Plot ownership

The caller owns figure construction, subplot geometry, scientific coordinates, labels, legends,
layout, saving, and closing. Matplotlib receives the immutable numerical arrays directly. Pi-PLS
provides no plotting submodule or hidden rendering layer.

For header-bearing CSV files, names can be retained with ordinary pandas operations:

```python
predictor_names = X.columns.astype(str).tolist()
response_names = Y.columns.astype(str).tolist()
```

For spectral line plots, use the physical coordinate in its existing order. Pi-PLS does not smooth,
interpolate, or reorder it. Do not move chart construction into an example support helper: the
maintained example should show which result fields and Matplotlib operations produce the figure.

## Interpretation boundary

Fitted factors, scores, loadings, coefficients, biplots, and observation diagnostics describe one
fitted model. Prediction diagnostics describe the prediction source stated by their provenance.
None of these displays supplies uncertainty intervals, causal interpretation, automatic variable
selection, or an unbiased post-selection performance estimate.
