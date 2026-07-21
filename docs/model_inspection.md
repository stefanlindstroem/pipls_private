# Numerical model inspection

`pipls.inspection` contains fitted-model analysis computations that do not depend on pandas or
Matplotlib. `pipls.plotting` consumes those immutable results and provides optional Matplotlib
figures. Dataset-specific tables, OOF orchestration, and multipage reports remain separate complete-example
workflows. For the shortest fitted-model path, begin with [`quickstart.md`](quickstart.md). The
numbered workflows are summarized in [`examples.md`](examples.md); their support modules are report
infrastructure, not prerequisites for ordinary estimator use.

## Analysis-model boundary

Example 09 explicitly fits Pi-PLS and ordinary PLS paths and plots both CV-MSE curves. Examples
10–12 evaluate only a Pi-PLS path before fitting
and inspecting one selected Pi-PLS model. Inspection of $P$, $D$, $Q$, and $QD$ remains explicitly
Pi-PLS-specific. Scores, loadings, coefficients, biplots, observation diagnostics, and prediction
diagnostics are shared PLS-family analyses with estimator-neutral names.

The shared numerical extraction accepts compatible fitted `PiPLSRegression` and scikit-learn
`PLSRegression` models through their public fitted attributes and transformation methods. The
numbered real-data analysis examples apply those tools only to the selected Pi-PLS model. Ordinary
PLS appears in the dedicated comparison example rather than as an automatic stage of normal use.

Import these names from the submodule:

```python
from pipls.inspection import (
    BiplotCoordinates,
    LatentStructure,
    ObservationDiagnostics,
    PiPLSDisplayFactors,
    PredictionDiagnostics,
    pipls_display_factors,
    biplot_coordinates,
    latent_structure,
    observation_diagnostics,
    prediction_diagnostics,
)
```

They are not top-level `pipls` exports.

## Pi-PLS display factors

A fitted `PiPLSRegression` exposes the centered/scaled regression map through
`model.decomposition_`:

\begin{equation}
B_{\mathrm{cs}} = P D Q^\mathsf{T}.
\end{equation}

Singular-vector signs are not unique. `pipls_display_factors()` copies $P$, $D$, and $Q$ and chooses
a deterministic display sign for every component. It finds the first largest-magnitude entry in
$P_{:k}$, makes that entry nonnegative, and applies the same sign to $Q_{:k}$. A zero predictor
direction is unchanged.

```python
from pipls.inspection import pipls_display_factors

model.fit(X, Y)
factors = pipls_display_factors(model.decomposition_)

P = factors.predictor_directions
d = factors.dilation
Q = factors.response_directions
QD = factors.weighted_response_directions
```

The copied factors preserve the fitted map:

\begin{equation}
P_{\mathrm{display}} D Q_{\mathrm{display}}^\mathsf{T}
=
P D Q^\mathsf{T}.
\end{equation}

`weighted_response_directions` stores $QD$, so column $k$ contains $d_kq_{:k}$. All returned arrays
are defensive read-only copies. The fitted estimator and its `decomposition_` arrays are not
changed.

Use **predictor rotation** or **predictor direction** for $P$. It is distinct from
`model.x_loadings_`, which is the score-reconstruction loading matrix.

## Prediction diagnostics

`prediction_diagnostics()` receives observed and predicted responses explicitly. It does not read
training data from an estimator and does not infer how predictions were obtained.

```python
from pipls.inspection import prediction_diagnostics

Y_pred = model.predict(X_test)
diagnostics = prediction_diagnostics(
    Y_test,
    Y_pred,
    prediction_kind="external test predictions",
)
```

Residuals use

\begin{equation}
e_{ij}=y_{ij}-\hat y_{ij}.
\end{equation}

For a common multivariate display, centers and sample standard deviations are estimated from the
supplied observed responses:

\begin{equation}
\bar y_j=\frac{1}{n}\sum_{i=1}^{n} y_{ij},
\qquad
s_j=\sqrt{\frac{1}{n-1}\sum_{i=1}^{n}(y_{ij}-\bar y_j)^2}.
\end{equation}

The same center and scale are applied to observations and predictions:

\begin{equation}
z_{ij}=\frac{y_{ij}-\bar y_j}{s_j},
\qquad
\hat z_{ij}=\frac{\hat y_{ij}-\bar y_j}{s_j},
\qquad
e^{(z)}_{ij}=z_{ij}-\hat z_{ij}.
\end{equation}

The result stores original and standardized arrays, response centers and scales, response-wise
standardized RMSE, and the prediction provenance. One-dimensional inputs are normalized to
`(n_samples, 1)` output arrays.

The accepted provenance labels are:

- `fitted values`;
- `fixed-parameter OOF predictions`;
- `selection-conditioned OOF predictions`;
- `external test predictions`.

The function rejects nonfinite values, shape disagreement, fewer than two observations, constant
response columns, and an unrecognized provenance label. The standardization is for diagnostics;
it does not modify the estimator or its predictions in original response units.

## Shared PLS-family latent structure

`latent_structure()` accepts a compatible fitted PLS-family model and copies only its public fitted
arrays. Both `PiPLSRegression` and scikit-learn `PLSRegression` satisfy the supported structural
contract:

```python
from pipls.inspection import latent_structure

structure = latent_structure(model)
```

The immutable result contains:

- `x_scores`, with shape `(n_samples, n_components)`;
- `x_loadings`, with shape `(n_features, n_components)`;
- `y_loadings`, with shape `(n_targets, n_components)`;
- `coefficients`, with shape `(n_targets, n_features)`.

The arrays are defensive read-only copies. No scores, loadings, or coefficients are recomputed,
rescaled, or sign-adjusted by `pipls`. They describe the supplied fitted model and are not validation
results. The function validates the public fitted attributes structurally; it does not require a
concrete estimator class.

## Shared PLS-family score-loading biplot

`biplot_coordinates()` constructs a two-component score-loading biplot from an existing
`LatentStructure`. For selected score and X-loading columns $t_k$ and $p_k$, it uses

\begin{equation}
a_k=\sqrt{\frac{\lVert p_k\rVert_2}{\lVert t_k\rVert_2}},\qquad
\tilde t_k=a_kt_k,\qquad
\tilde p_k=\frac{p_k}{a_k}.
\end{equation}

The scaling balances the Euclidean norms and preserves the selected reconstruction:

\begin{equation}
\tilde T\tilde P^\mathsf{T}=T_{\mathcal K}P_{\mathcal K}^\mathsf{T}.
\end{equation}

```python
from pipls.inspection import biplot_coordinates
from pipls.plotting import plot_biplot

coordinates = biplot_coordinates(structure, components=(0, 1))
figure, axis = plot_biplot(
    coordinates,
    predictor_names=predictor_names,
)
```

The figure shows sample coordinates and X-variable arrows only. It does not add Y-variable arrows,
confidence regions, inferred groups, or automatic importance claims. The maintained demonstration is
restricted to Pulp, where fourteen predictor arrows remain readable. Sugarcane and Tobacco do not
generate biplots because their spectral predictor counts make such a display unsuitable.

## Shared PLS-family observation diagnostics

`observation_diagnostics()` accepts a compatible fitted model and explicit predictor observations:

```python
from pipls.inspection import observation_diagnostics

observation_result = observation_diagnostics(model, X)
```

The model must expose fitted `x_scores_` and `x_loadings_` arrays together with public
`transform()` and `inverse_transform()` methods. Let $t_i$ be the transformed score of supplied
observation $i$, and let $\bar t_{\mathrm{train}}$ and $S_{T,\mathrm{train}}$ be the center and sample
covariance of the fitted training scores. The squared score distance is

\begin{equation}
h_i=(t_i-\bar t_{\mathrm{train}})^\mathsf{T}S_{T,\mathrm{train}}^{+}(t_i-\bar t_{\mathrm{train}}).
\end{equation}

Here $+$ denotes the Moore--Penrose inverse. The row-wise squared X-reconstruction residual is

\begin{equation}
q_i=\lVert x_i-\hat x_i\rVert_2^2.
\end{equation}

The reconstruction $\hat x_i$ is obtained from the fitted model's public
transform/inverse-transform round trip. The immutable result contains `score_distance` and
`x_reconstruction_residual`. These are raw descriptive quantities; the package does not add
theoretical limits, automatic outlier labels, or contribution diagnostics.

## Interpretation boundary

Display factors describe a fixed fitted model. They are not validation results. Prediction
statistics mean different things depending on their provenance. In particular,
`selection-conditioned OOF predictions` indicate that the same observations contributed to an
earlier parameter-selection path, so they are not an independent post-selection performance
estimate.

## Supplying scientific variable names

Variable-name acquisition is outside the plotting API. Plotting functions receive labels explicitly
and do not read files, inspect pandas objects, or invent scientific meanings for unlabeled arrays.
When a CSV file stores variable names in its header, an example can preserve them directly:

```python
import pandas as pd

X = pd.read_csv("X.csv")
Y = pd.read_csv("Y.csv")
predictor_names = X.columns.astype(str).tolist()
response_names = Y.columns.astype(str).tolist()
```

Those lists can then be passed to Pi-PLS-specific factor plots and shared PLS-family plots. This is an
example-level I/O operation, not a package requirement. A user whose arrays do not come from
header-bearing files can supply names from a schema, laboratory-information system, domain
metadata, or an explicit list:

```python
predictor_names = ["Temperature", "Pressure", "Flow rate"]
response_names = ["Yield", "Purity"]
```

For spectral line plots, CSV headers may instead be converted to the physical coordinate, for
example `X.columns.to_numpy(dtype=float)`, while response names still come from `Y.columns` or
another metadata source.

## Plotting computed results

Install the optional plotting dependency with:

```bash
python -m pip install "pipls[plot]"
```

The plotting names remain in their own submodule:

```python
from pipls.plotting import (
    plot_biplot,
    plot_coefficients,
    plot_observation_diagnostics,
    plot_pipls_dilation,
    plot_pipls_predictor_directions,
    plot_pipls_response_directions,
    plot_pipls_weighted_response_directions,
    plot_observed_vs_predicted,
    plot_residuals_vs_predicted,
    plot_standardized_rmse,
    plot_scores,
    plot_x_loadings,
    plot_y_loadings,
)
```

The Pi-PLS factorization is $P D Q^{\mathsf T}$. Four separate plotting functions display the
predictor directions $P$, dilation values $d_k=D_{kk}$, response directions $Q$, and weighted
response directions $QD$. Each function draws one chart on one axis. The caller decides whether to
use them separately or in a panel:

```python
import matplotlib.pyplot as plt

factor_figure, factor_axes = plt.subplots(2, 2, figsize=(12, 9), layout="constrained")
plot_pipls_predictor_directions(
    factors,
    predictor_style="bar",
    predictor_names=feature_names,
    components=[0, 1],
    ax=factor_axes[0, 0],
)
plot_pipls_dilation(factors, components=[0, 1], ax=factor_axes[0, 1])
plot_pipls_response_directions(
    factors,
    response_names=target_names,
    components=[0, 1],
    ax=factor_axes[1, 0],
)
plot_pipls_weighted_response_directions(
    factors,
    response_names=target_names,
    components=[0, 1],
    ax=factor_axes[1, 1],
)
for axis in (factor_axes[0, 0], factor_axes[1, 0], factor_axes[1, 1]):
    axis.legend(title="Component")
factor_figure.suptitle("Pi-PLS factors")
```

For an ordered physical predictor coordinate, use line rendering for $P$:

```python
predictor_figure, predictor_axis = plot_pipls_predictor_directions(
    factors,
    predictor_style="line",
    predictor_axis=wavelength_nm,
    predictor_axis_label="Wavelength (nm)",
)
predictor_axis.legend(title="Component")
```

The supplied coordinate order is preserved, including decreasing wavenumber axes. The function does
not smooth, interpolate, reorder, or infer a spectral representation. Categorical plots require the
caller to supply predictor or response names.

Three separate functions display prediction diagnostics: standardized observed versus predicted
responses, standardized residuals versus standardized predictions, and response-wise standardized
RMSE. Each can create a standalone chart or draw into a caller-owned panel:

```python
import matplotlib.pyplot as plt

prediction_figure, prediction_axes = plt.subplots(
    1, 3, figsize=(13, 4.2), layout="constrained"
)
plot_observed_vs_predicted(
    diagnostics,
    response_names=target_names,
    responses=[0, 2],
    include_prediction_kind=False,
    ax=prediction_axes[0],
)
plot_residuals_vs_predicted(
    diagnostics,
    response_names=target_names,
    responses=[0, 2],
    include_prediction_kind=False,
    ax=prediction_axes[1],
)
plot_standardized_rmse(
    diagnostics,
    response_names=target_names,
    responses=[0, 2],
    include_prediction_kind=False,
    ax=prediction_axes[2],
)
prediction_axes[0].legend()
prediction_axes[1].legend()
prediction_figure.suptitle(diagnostics.prediction_kind)
```

Standalone calls include `diagnostics.prediction_kind` in the axis title by default. In a panel,
set `include_prediction_kind=False` and report provenance once at figure level. The functions do not
create legends, panels, or files.

Raw PLS-family observation diagnostics use a separate result object:

```python
observation_figure, observation_axis = plot_observation_diagnostics(
    observation_diagnostics,
)
```

The function returns the figure and its single axis and adds neither theoretical limits nor
automatic observation labels.

### Standalone and panel composition

The single-chart functions accept `ax=None`. Without an axis they create one standalone figure. A
caller that wants a panel creates the layout and passes its axes explicitly:

```python
import matplotlib.pyplot as plt

figure, axes = plt.subplots(1, 2, figsize=(12, 5), layout="constrained")

plot_scores(structure, components=(0, 1), ax=axes[0])
plot_x_loadings(
    structure,
    predictor_style="bar",
    predictor_names=feature_names,
    components=[0, 1],
    ax=axes[1],
)
axes[1].legend(title="Component")
figure.savefig("latent_structure.pdf")
```

A supplied axis is not cleared, resized, or placed into a new figure. `figsize` therefore applies
only when the plotting function creates the figure. The functions provide concise semantic axis
labels and titles; callers may replace them through ordinary Matplotlib methods. Multi-series
artists carry labels, but legend creation, placement, and styling remain caller-owned.

### Shared PLS-family figures

The shared plotting functions consume `LatentStructure`, not an estimator:

```python
score_figure, score_axis = plot_scores(
    structure,
    components=(0, 1),
)

x_loading_figure, x_loading_axis = plot_x_loadings(
    structure,
    predictor_style="bar",
    predictor_names=feature_names,
    components=[0, 1],
)

y_loading_figure, y_loading_axis = plot_y_loadings(
    structure,
    response_names=target_names,
    components=[0, 1],
)

coefficient_figure, coefficient_axis = plot_coefficients(
    structure,
    predictor_style="bar",
    predictor_names=feature_names,
    response_names=target_names,
    responses=[0, 2],
)
```

`plot_scores()` requires exactly two distinct zero-based component indices. Optional sample
labels annotate observations, but the function does not infer groups or draw confidence regions.
X loadings and response-specific coefficients share the explicit bar-versus-line predictor
contract. Selected X-loading components use grouped bars or overlaid lines on one axis. Selected
Y-loading components use grouped bars on one response axis. Selected coefficient responses use
grouped bars or overlaid lines on one predictor axis. Categorical displays require caller-supplied
predictor or response names. For line rendering, the caller supplies the physical predictor
coordinate and axis label, whose order is preserved.

The shared plotting surface also includes `plot_observation_diagnostics()`, which draws
raw score distance against squared X-reconstruction residual. It does not include theoretical
limits, automatic observation labels, or contribution plots. Biplots, confidence ellipses, VIP,
automatic variable selection, uncertainty intervals, and permutation tests remain outside the
implemented surface.

The [Pulp tutorial](tutorials/pulp.md#interpret-the-fitted-model) demonstrates these inspection and
plotting APIs one figure at a time. The complete workflow retains example-owned I/O. It derives predictor and
response names visibly from the Pulp CSV headers, reads fixed component choices from the canonical
path artifacts, and clones the fixed Pi-PLS estimator
inside the same five non-shuffled folds, writes seven long-form CSV files under
`examples/results/pulp_post_analysis/`, rereads them, and constructs one multipage PDF. The example
composer creates a $P$/$D$/$Q$/$QD$ factor page, a prediction-diagnostic page, and a $2\times2$
latent-model page containing scores, the balanced biplot, X loadings, and Y loadings. The OOF
predictions are labeled `selection-conditioned OOF predictions` because the fixed parameters were
chosen after examining paths computed from the same observations.

The standardized response columns in `predictions.csv` use centers and sample standard deviations
computed from the complete supplied observed-response matrix for display. They are not the
fold-local response scales used by the component-path scorer, so their aggregate squared values
should not be expected to equal the reported mean fold CV-MSE exactly.

The Sugarcane workflow described in [`examples.md`](examples.md#complete-pi-pls-analyses) uses the
same seven-table contract under `examples/results/sugarcane_post_analysis/`. The script reads the
physical wavelength coordinate from the numeric `X.csv` headers and passes it to the report renderer
with the label `Wavelength (nm)`. The
canonical loading, direction, and coefficient values remain in long-form CSV tables; the report
uses line rendering only because the example supplies the ordered physical coordinate explicitly.
Its shared latent-model page places scores, spectral X loadings, and Y loadings in a $1\times3$
panel. All four response headers (`TS`, `CP`, `ADF`, and `IVOMD`) are retained.

The Tobacco workflow described in [`examples.md`](examples.md#complete-pi-pls-analyses) preserves
the strictly decreasing wavenumber coordinate read from the `X.csv` headers and uses it for Pi-PLS
predictor directions, Pi-PLS X loadings, and response-specific coefficient curves. All
thirteen response headers are partitioned into deterministic source-order pages of at most five
responses; the report renderer rejects pagination that omits, duplicates, or reorders a response.
The Pulp report derives its biplot from the canonical `x_scores.csv` and `x_loadings.csv`; no additional biplot table is required.

Tobacco writes the seven common post-analysis tables plus
`observation_diagnostics.csv`, whose columns are `sample`, `score_distance`, and
`x_reconstruction_residual`. Its shared latent-model page combines scores, spectral X loadings, Y
loadings, and these raw observation diagnostics in a $2\times2$ panel without theoretical outlier
limits. Coefficient curves remain on full-width response pages in all three reports.
