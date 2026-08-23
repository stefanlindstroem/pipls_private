# Model inspection concepts

`pipls.inspection` computes immutable numerical results from fitted models or explicit predictions.
Π-PLS-specific inspection covers $\mathbf{P}$, $\mathbf{D}$, $\mathbf{Q}$, and
$\mathbf{Q}\mathbf{D}$; scores, loadings, coefficients, biplots,
observation diagnostics, and prediction diagnostics use estimator-neutral PLS-family objects.
Maintained examples render these arrays directly, but rendering is not part of the numerical API.
Inspection calculations use range-safe scaled operations where ordinary norms, covariance products,
sample scales, or squared residuals could overflow. A derived quantity that cannot be represented
as finite float64 raises `ValueError` rather than entering an immutable result.

The [Pulp tutorial](tutorials/pulp.md#interpret-representative-fitted-model-plots) shows a
representative subset. The generated [inspection API](api/inspection.md) documents exact fields and
signatures.

## Numerical results

### Π-PLS display factors

A fitted `PiPLSRegression` stores the centered and scaled regression map as

\begin{equation}
\mathbf{B}_{\mathrm{cs}}=\mathbf{P}\mathbf{D}\mathbf{Q}^{\mathsf T}.
\end{equation}

This is the method's [diagonal latent coupling](theory.md#diagonal-latent-coupling).
`pipls_display_factors()` returns defensive read-only copies and applies one deterministic display
sign per paired latent mode. The same sign is applied to the paired columns of $\mathbf{P}$
and $\mathbf{Q}$, so the regression map is unchanged. By default, the first largest-magnitude
predictor entry is made
nonnegative.

```python
from pipls.inspection import pipls_display_factors

factors = pipls_display_factors(model.decomposition_)
```

A response can instead define the display orientation. The caller resolves a scientific response
name to its zero-based row because the decomposition intentionally contains no labels:

```python
ti_index = response_names.index("TI")
factors = pipls_display_factors(
    model.decomposition_,
    response_index=ti_index,
    response_sign="positive",
)
```

This makes the selected response entry nonnegative in every component. `response_sign="negative"`
requests nonpositive entries. If the selected response entry is exactly zero, that component falls
back to the default predictor-based convention. These choices orient an equivalent factorization;
they do not change predictions or establish that the selected response has a positive or negative
physical effect.

$\mathbf{P}$ contains orthonormal predictor directions, not X loadings. X loadings belong to score
reconstruction and are stored in `LatentStructure`.

### Latent structure

`latent_structure()` copies the fitted PLS-family scores, X loadings, Y loadings, and original-unit
coefficient matrix into a read-only `LatentStructure`. It does not recompute, rescale, or sign-adjust
them. They describe the supplied full-data fit and are not validation results.

### Biplot coordinates

`biplot_coordinates()` balances two score and X-loading columns while preserving their rank-two
reconstruction. For component $k$ it uses

\begin{equation}
a_k=\sqrt{\frac{\lVert p_k\rVert_2}{\lVert t_k\rVert_2}},\qquad
\widetilde t_k=a_kt_k,\qquad
\widetilde p_k=\frac{p_k}{a_k}.
\end{equation}

The result contains sample coordinates and predictor-arrow coordinates. It does not infer groups,
confidence regions, or variable importance.

### Prediction diagnostics

`prediction_diagnostics()` receives observed and predicted responses explicitly.
`PredictionDiagnostics` accepts those independent arrays and the required provenance label, then
derives the residuals, response centers and scales, response-standardized arrays, response-wise
standardized RMSE, and response-wise coefficient of determination once. Display standardization
uses the supplied observed responses and does not alter predictions in original units.

Selection-conditioned OOF predictions are descriptive post-selection diagnostics, not an
independent estimate of future performance.

### Observation diagnostics

`observation_diagnostics()` returns squared score distance from the training-score center and
squared X-reconstruction residual from the model's public transform/inverse-transform round trip.
These are raw descriptive quantities; the package supplies no theoretical limits, automatic
outlier labels, or contribution diagnostics.

## Quantity catalogue

| Question | Numerical field | Interpretation and usual display |
|---|---|---|
| Where are samples in the latent plane? | `structure.x_scores` | Scatter two components; proximity means similar displayed score coordinates |
| How do predictors reconstruct scores? | `structure.x_loadings` | Plot or group selected loading columns; these are not regression coefficients |
| How do responses enter the latent representation? | `structure.y_loadings` | Compare selected loading columns across named responses |
| What are the Π-PLS predictor directions? | `factors.predictor_directions` | Plot columns of $\mathbf{P}$ against names or a physical predictor coordinate |
| How strong is each paired latent mode? | `factors.dilation` | Compare the nonnegative dilations $D_k=D_{kk}$ |
| What are the Π-PLS response directions? | `factors.response_directions` | Compare columns of $\mathbf{Q}$ across responses |
| What is each response mode after dilation? | `factors.weighted_response_directions` | Compare columns of $\mathbf{Q}\mathbf{D}$ across responses |
| What is the original-unit linear map? | `structure.coefficients` | Plot one coefficient row per response, respecting variable units |
| Which observations are distant or poorly reconstructed? | `observations.score_distance`, `observations.x_reconstruction_residual` | Scatter the two raw diagnostics |
| How well do predictions agree with observations? | `PredictionDiagnostics` arrays | Use observed/predicted, residual, response-wise RMSE, and response-wise $R^2$ views with provenance shown |

### Scores { #scores }

Use two columns of `structure.x_scores`. Proximity means similar coordinates in the displayed latent
plane. The display is exploratory and does not establish groups, confidence regions, or outliers.

### Score-loading biplot { #score-loading-biplot }

Plot `BiplotCoordinates.sample_coordinates` and draw vectors from the origin to
`predictor_coordinates`. Similar arrow directions indicate similar loading patterns in the displayed
plane; they are not causal effects or automatic importance measures. Optional `textalloc` may
reposition predictor labels after limits, aspect, titles, and legends are final. The maintained Pulp biplot supplies the predictor-arrow shafts as line obstacles and lets the
allocator avoid other predictor labels; placement remains a rendering heuristic.

### X loadings { #x-loadings }

Use selected columns of `structure.x_loadings`. Large absolute values indicate strong participation
in score reconstruction. X loadings are not regression coefficients and do not by themselves
measure predictive importance. Component signs may reverse without changing the model.

### Y loadings { #y-loadings }

Use selected columns of `structure.y_loadings` with explicit response labels. Responses with similar
patterns are represented similarly across the selected components. With response scaling, these
loadings describe the standardized fitted representation rather than original-unit response values.

### Predictor directions $\mathbf{P}$ { #predictor-directions }

Use columns of `factors.predictor_directions`. They are orthonormal predictor directions. The
corresponding columns of $\mathbf{Q}$ are orthonormal response directions, and $\mathbf{D}$ pairs
and scales the two sets of directions in $\mathbf{P}\mathbf{D}\mathbf{Q}^{\mathsf T}$. Predictor
directions are distinct from X loadings because they belong to the regression factorization rather
than score reconstruction.

Theory: [Diagonal latent coupling](theory.md#diagonal-latent-coupling).

### Dilation $\mathbf{D}$ { #dilation }

Each value $D_k=D_{kk}$ is the dilation of paired latent mode $k$ and should be interpreted
together with the matching columns of $\mathbf{P}$ and $\mathbf{Q}$.

### Response directions $\mathbf{Q}$ { #response-directions }

The columns of `factors.response_directions` are the orthonormal response directions of the paired
latent modes before dilation. Use explicit response labels when comparing them.

### Weighted response directions $\mathbf{Q}\mathbf{D}$ { #weighted-response-directions }

`factors.weighted_response_directions` is a derived read-only array that combines response-side
orientation and mode strength. Column $k$ is $D_kQ_{:k}$. Compare it with $\mathbf{Q}$ when
distinguishing direction from scaled contribution to the centered/scaled regression map.

### Regression coefficients { #regression-coefficients }

Each row of `structure.coefficients` describes one response in original predictor and response
units. Raw magnitudes are not directly comparable across variables with different units, and
correlated predictors may share coefficient weight.

### Observed versus predicted { #observed-versus-predicted }

Use `observed_standardized` and `predicted_standardized` with an identity reference. Keep
`prediction_kind` visible so fitted, OOF, and external-test predictions are not conflated.

### Residuals versus predicted { #residuals-versus-predicted }

Use `predicted_standardized` against `residual_standardized` with a zero-residual reference.
Patterns may reveal response-dependent scale or systematic error, but they do not supply formal
uncertainty tests.

### Standardized RMSE { #standardized-rmse }

`standardized_rmse` compares response-wise error with each displayed response's observed spread.
Lower values mean smaller relative error. These values are not generally equal to the mean
fold-local standardized loss used during path selection.

### Response-wise coefficient of determination { #response-r2 }

`response_r2` is calculated independently for each response from the supplied predictions:

\[
R_j^2 = 1 - \frac{\sum_i (y_{ij} - \hat y_{ij})^2}{\sum_i (y_{ij} - \bar y_j)^2}.
\]

The interpretation follows `prediction_kind`. For fitted values it is a descriptive training-fit
quantity; for OOF or external-test predictions it describes those supplied predictions. It is not
by itself evidence of unbiased post-selection performance.

### Observation diagnostics { #observation-diagnostics }

Plot `score_distance` against `x_reconstruction_residual`. Large values identify observations that
are distant in fitted score space, poorly reconstructed in predictor space, or both. No automatic
threshold is implied.

## Rendering and metadata

The caller owns chart composition, scientific coordinates, labels, legends, layout, saving, and
closing. Predictor and response names may come from ordinary Python sequences or data-frame column
labels. For spectral plots, retain the physical coordinate in its existing order; Π-PLS does not
smooth, interpolate, or reorder it.

## Interpretation boundary

Fitted factors, scores, loadings, coefficients, biplots, and observation diagnostics describe one
fitted model. Prediction diagnostics describe the prediction source stated by their provenance.
None of these quantities supplies uncertainty intervals, causal interpretation, automatic variable
selection, or an unbiased post-selection performance estimate.
