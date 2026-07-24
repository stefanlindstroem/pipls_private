# Model inspection concepts

`pipls.inspection` computes immutable numerical results from fitted models or explicit predictions.
Pi-PLS-specific inspection covers $P$, $D$, $Q$, and $QD$; scores, loadings, coefficients, biplots,
observation diagnostics, and prediction diagnostics use estimator-neutral PLS-family objects.
Maintained examples render these arrays directly, but rendering is not part of the numerical API.
Inspection calculations use range-safe scaled operations where ordinary norms, covariance products,
sample scales, or squared residuals could overflow. A derived quantity that cannot be represented
as finite float64 raises `ValueError` rather than entering an immutable result.

The [Pulp tutorial](tutorials/pulp.md#interpret-representative-fitted-model-plots) shows a
representative subset. The generated [inspection API](api/inspection.md) documents exact fields and
signatures.

## Numerical results

### Pi-PLS display factors

A fitted `PiPLSRegression` stores the centered and scaled regression map as

\begin{equation}
B_{\mathrm{cs}}=PDQ^{\mathsf T}.
\end{equation}

`pipls_display_factors()` returns defensive read-only copies and applies one deterministic display
sign per component. The same sign is applied to the paired columns of $P$ and $Q$, so the regression
map is unchanged.

```python
from pipls.inspection import pipls_display_factors

factors = pipls_display_factors(model.decomposition_)
```

$P$ contains predictor rotations, not X loadings. X loadings belong to score reconstruction and are
stored in `LatentStructure`.

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

`prediction_diagnostics()` receives observed and predicted responses explicitly. It stores original
and response-standardized observations, predictions, residuals, response centers and scales,
response-wise standardized RMSE, and a required provenance label. Display standardization uses the
supplied observed responses and does not alter predictions in original units.

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
| What are the Pi-PLS predictor modes? | `factors.predictor_directions` | Plot columns of $P$ against names or a physical predictor coordinate |
| How strong is each paired mode? | `factors.dilation` | Compare the nonnegative diagonal values of $D$ |
| What are the response-side modes? | `factors.response_directions` | Compare columns of $Q$ across responses |
| What is each response mode after dilation? | `factors.weighted_response_directions` | Compare columns of $QD$ across responses |
| What is the original-unit linear map? | `structure.coefficients` | Plot one coefficient row per response, respecting variable units |
| Which observations are distant or poorly reconstructed? | `observations.score_distance`, `observations.x_reconstruction_residual` | Scatter the two raw diagnostics |
| How well do predictions agree with observations? | `PredictionDiagnostics` arrays | Use observed/predicted, residual, and response-wise RMSE views with provenance shown |

### Scores { #scores }

Use two columns of `structure.x_scores`. Proximity means similar coordinates in the displayed latent
plane. The display is exploratory and does not establish groups, confidence regions, or outliers.

### Score-loading biplot { #score-loading-biplot }

Plot `BiplotCoordinates.sample_coordinates` and draw vectors from the origin to
`predictor_coordinates`. Similar arrow directions indicate similar loading patterns in the displayed
plane; they are not causal effects or automatic importance measures. Optional
`adjustText.adjust_text()` may reposition predictor labels after limits, aspect, titles, and legends
are final, but its placement is heuristic.

### X loadings { #x-loadings }

Use selected columns of `structure.x_loadings`. Large absolute values indicate strong participation
in score reconstruction. X loadings are not regression coefficients and do not by themselves
measure predictive importance. Component signs may reverse without changing the model.

### Y loadings { #y-loadings }

Use selected columns of `structure.y_loadings` with explicit response labels. Responses with similar
patterns are represented similarly across the selected components. With response scaling, these
loadings describe the standardized fitted representation rather than original-unit response values.

### Predictor directions $P$ { #predictor-directions }

Use columns of `factors.predictor_directions`. They define predictor rotations paired with response
rotations in $PDQ^{\mathsf T}$ and are distinct from X loadings because they belong to the
regression factorization rather than score reconstruction.

Theory: [Diagonal latent coupling](theory.md#diagonal-latent-coupling).

### Dilation $D$ { #dilation }

Each value $d_k=D_{kk}$ scales one paired predictor-response mode and should be interpreted together
with the matching columns of $P$ and $Q$.

### Response directions $Q$ { #response-directions }

The columns of `factors.response_directions` describe the response rotation of each paired mode
before dilation. Use explicit response labels when comparing them.

### Weighted response directions $QD$ { #weighted-response-directions }

`factors.weighted_response_directions` combines response-side orientation and mode strength through
$d_kq_{:k}$. Compare it with $Q$ when distinguishing direction from scaled contribution to the
centered/scaled regression map.

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

### Observation diagnostics { #observation-diagnostics }

Plot `score_distance` against `x_reconstruction_residual`. Large values identify observations that
are distant in fitted score space, poorly reconstructed in predictor space, or both. No automatic
threshold is implied.

## Rendering and metadata

The caller owns chart composition, scientific coordinates, labels, legends, layout, saving, and
closing. Predictor and response names may come from ordinary Python sequences or data-frame column
labels. For spectral plots, retain the physical coordinate in its existing order; Pi-PLS does not
smooth, interpolate, or reorder it.

## Interpretation boundary

Fitted factors, scores, loadings, coefficients, biplots, and observation diagnostics describe one
fitted model. Prediction diagnostics describe the prediction source stated by their provenance.
None of these quantities supplies uncertainty intervals, causal interpretation, automatic variable
selection, or an unbiased post-selection performance estimate.
