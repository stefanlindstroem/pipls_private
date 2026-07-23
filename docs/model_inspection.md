# Model inspection

`pipls.inspection` computes immutable numerical results from fitted models or explicit predictions.
`pipls.plotting` renders those results without changing the estimator. This page defines the full
inspection and plotting catalogue. The [Pulp tutorial](tutorials/pulp.md#interpret-representative-fitted-model-plots)
uses a small representative subset in one real-data analysis; the
[plotting API](api/plotting.md) gives exact signatures.

Pi-PLS-specific inspection covers $P$, $D$, $Q$, and $QD$. Scores, loadings, coefficients, biplots,
observation diagnostics, and prediction diagnostics use estimator-neutral PLS-family objects.

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

The supported provenance labels are:

- `fitted values`;
- `fixed-parameter OOF predictions`;
- `selection-conditioned OOF predictions`;
- `external test predictions`.

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

## Interpreting plots

### Scores { #scores }

`plot_scores()` shows samples in two selected predictor-score coordinates. Proximity means similar
coordinates in that displayed latent plane; separation means differences represented by those two
components. The plot is exploratory and does not establish groups, confidence regions, or
outliers.

API: [`plot_scores()`](api/plotting.md#pipls.plotting.plot_scores).

### Score-loading biplot { #score-loading-biplot }

`biplot_coordinates()` calculates balanced sample and predictor coordinates for two components.
Plot `sample_coordinates` with `Axes.scatter()`, draw vectors from the origin to
`predictor_coordinates`, and create predictor labels with `Axes.text()`. For dense labels,
`adjustText.adjust_text()` can move those text artists after titles, limits, aspect, and legends have
been configured. Automatic placement is heuristic, so unusually dense diagrams may still need
manual adjustment.

Similar arrow directions indicate similar loading patterns in the displayed plane. A sample lying
in an arrow's direction has a positive coordinate along that displayed predictor direction. These
are geometric statements, not causal effects or automatic importance measures.

API: [`biplot_coordinates()`](api/inspection.md#pipls.inspection.biplot_coordinates).

### X loadings { #x-loadings }

`plot_x_loadings()` shows how original predictors reconstruct the selected predictor scores. Large
absolute values indicate strong participation in that reconstruction. X loadings are not regression
coefficients and do not by themselves measure predictive importance. Component signs may reverse
without changing the model.

API: [`plot_x_loadings()`](api/plotting.md#pipls.plotting.plot_x_loadings).

### Y loadings { #y-loadings }

`plot_y_loadings()` shows how responses participate in the response-side latent representation.
Responses with similar patterns are represented similarly across the selected components. When the
model scales responses, the loadings describe the standardized fitted representation rather than
response values in original units.

API: [`plot_y_loadings()`](api/plotting.md#pipls.plotting.plot_y_loadings).

### Predictor directions $P$ { #predictor-directions }

`plot_pipls_predictor_directions()` shows the columns of $P$. These predictor rotations define the
directions paired with the response rotations in $PDQ^{\mathsf T}$. They are distinct from X
loadings because they belong to the regression factorization rather than score reconstruction.

API: [`plot_pipls_predictor_directions()`](api/plotting.md#pipls.plotting.plot_pipls_predictor_directions).
Theory: [Diagonal latent coupling](theory.md#diagonal-latent-coupling).

### Dilation $D$ { #dilation }

`plot_pipls_dilation()` shows $d_k=D_{kk}$. Each value scales one paired predictor-response mode.
Dilation should be interpreted together with the matching columns of $P$ and $Q$; it is not a
standalone variable-importance score.

API: [`plot_pipls_dilation()`](api/plotting.md#pipls.plotting.plot_pipls_dilation).

### Response directions $Q$ { #response-directions }

`plot_pipls_response_directions()` shows the response rotation of each paired mode before dilation.
It describes response-side orientation. A pronounced direction can still have a modest contribution
when its dilation is small.

API: [`plot_pipls_response_directions()`](api/plotting.md#pipls.plotting.plot_pipls_response_directions).

### Weighted response directions $QD$ { #weighted-response-directions }

`plot_pipls_weighted_response_directions()` multiplies each response direction by its dilation. It
therefore combines response-side orientation and mode strength. Use $Q$ when direction alone is the
question and $QD$ when the scaled response-side contribution is the question.

API: [`plot_pipls_weighted_response_directions()`](api/plotting.md#pipls.plotting.plot_pipls_weighted_response_directions).

### Regression coefficients { #regression-coefficients }

`plot_coefficients()` shows the fitted linear map in original predictor and response units. A
coefficient is the modeled response change associated with one predictor-unit change while the
other predictors are held fixed in the linear model. Raw magnitudes are not directly comparable
across variables with different units, and correlated predictors may share coefficient weight.

API: [`plot_coefficients()`](api/plotting.md#pipls.plotting.plot_coefficients).

### Observed versus predicted { #observed-versus-predicted }

`plot_observed_vs_predicted()` standardizes each displayed response and compares observations with
predictions relative to the identity line. The provenance label determines whether the plot shows
fitted values, fixed-parameter OOF predictions, selection-conditioned OOF predictions, or external
test predictions.

API: [`plot_observed_vs_predicted()`](api/plotting.md#pipls.plotting.plot_observed_vs_predicted).

### Residuals versus predicted { #residuals-versus-predicted }

`plot_residuals_vs_predicted()` places standardized residuals against standardized predictions.
Curvature, changing spread, or response-specific bands can indicate structure not captured by the
linear model. The plot alone does not justify labeling individual samples as anomalous.

API: [`plot_residuals_vs_predicted()`](api/plotting.md#pipls.plotting.plot_residuals_vs_predicted).

### Standardized RMSE { #standardized-rmse }

`plot_standardized_rmse()` compares response-wise RMSE after division by the sample standard
deviation of the supplied observed response. Lower values mean smaller error relative to that
response's observed spread. These display values are not generally equal to the mean fold-local
standardized loss used during path selection.

API: [`plot_standardized_rmse()`](api/plotting.md#pipls.plotting.plot_standardized_rmse).

### Observation diagnostics { #observation-diagnostics }

`plot_observation_diagnostics()` places squared score distance against squared X-reconstruction
residual. Large values identify observations that are distant in the fitted score space, poorly
reconstructed in predictor space, or both. The package supplies no automatic thresholds or labels.

API: [`plot_observation_diagnostics()`](api/plotting.md#pipls.plotting.plot_observation_diagnostics).

## Plotting contract

Every public plotting function renders one chart on one Matplotlib `Axes` and returns
`(figure, axis)`. With `ax=None`, it creates one standalone figure. With a supplied axis, it draws on
that axis without clearing it or altering the surrounding figure. Plotters provide concise semantic
axis labels and titles, but callers may replace them. Multi-series artists are labeled; callers
create and style legends.

```python
import matplotlib.pyplot as plt

figure, axes = plt.subplots(1, 2, figsize=(12, 5), layout="constrained")
plot_scores(structure, components=(0, 1), ax=axes[0])
plot_x_loadings(
    structure,
    predictor_style="bar",
    predictor_names=predictor_names,
    components=[0, 1],
    ax=axes[1],
)
axes[1].legend(title="Component")
figure.savefig("latent_structure.pdf")
```

The caller owns subplot geometry, legends, figure-level titles, layout, display, saving, and closing.
See the [plotting API](api/plotting.md) for selection and labeling parameters.

## Scientific labels and coordinates

Plotting functions receive predictor names, response names, sample labels, and physical predictor
coordinates explicitly. They do not read files or infer scientific meaning. For header-bearing
CSV files, names can be retained with ordinary pandas operations:

```python
predictor_names = X.columns.astype(str).tolist()
response_names = Y.columns.astype(str).tolist()
```

For spectral line plots, pass the physical coordinate and its label explicitly. The plotting
functions preserve the supplied order and do not smooth, interpolate, or reorder it.

## Interpretation boundary

Fitted factors, scores, loadings, coefficients, biplots, and observation diagnostics describe one
fitted model. Prediction diagnostics describe the prediction source stated by their provenance.
None of these displays supplies uncertainty intervals, causal interpretation, automatic variable
selection, or an unbiased post-selection performance estimate.
