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
figure, axes = plot_biplot(
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
    plot_pipls_decomposition,
    plot_biplot,
    plot_coefficients,
    plot_observation_diagnostics,
    plot_scores,
    plot_x_loadings,
    plot_y_loadings,
    plot_prediction_diagnostics,
)
```

`plot_pipls_decomposition()` places all requested zero-based components on three shared axes:
predictor directions $P_{:k}$, weighted response directions $d_kq_{:k}$, and dilation values. For a
small scalar predictor set, component bars are grouped side by side within each named variable:

```python
factor_figure, factor_axes = plot_pipls_decomposition(
    factors,
    predictor_style="bar",
    predictor_names=feature_names,
    response_names=target_names,
    components=[0, 1],
)
```

For an ordered physical coordinate, choose line rendering explicitly:

```python
factor_figure, factor_axes = plot_pipls_decomposition(
    factors,
    predictor_style="line",
    predictor_axis=wavelength_nm,
    predictor_axis_label="Wavelength (nm)",
    response_names=target_names,
)
```

The supplied coordinate order is preserved, including decreasing wavenumber axes, and selected
components are overlaid as separate labeled lines. The function does not smooth, interpolate,
reorder, or infer a spectral representation.

Categorical plots require the caller to supply predictor and response names. The Pulp, Sugarcane,
and Tobacco post-analysis examples read those names from `X.csv` and `Y.csv`; the package itself
remains agnostic about whether labels originated in file headers or another metadata source.

`plot_prediction_diagnostics()` renders standardized observed versus predicted responses,
standardized residuals versus standardized predictions, and response-wise standardized RMSE:

```python
prediction_figure, prediction_axes = plot_prediction_diagnostics(
    diagnostics,
    response_names=target_names,
    responses=[0, 2],
)
```

The prediction provenance stored in `diagnostics.prediction_kind` is always included in the figure
title. Both plotting functions return `(figure, axes)`, where `axes` is a dictionary with stable
semantic names. They do not call `show()`, save files, retain estimators, or modify supplied arrays.
Importing `pipls` or `pipls.plotting` does not import Matplotlib; Matplotlib is loaded only when a
plotting function is called.

Raw PLS-family observation diagnostics use a separate result object:

```python
observation_figure, observation_axes = plot_observation_diagnostics(
    observation_diagnostics,
)
```

The function returns one `observation_diagnostics` axis and adds neither theoretical limits nor
automatic observation labels.


### Shared PLS-family figures

The shared plotting functions consume `LatentStructure`, not an estimator:

```python
score_figure, score_axes = plot_scores(
    structure,
    components=(0, 1),
)

x_loading_figure, x_loading_axes = plot_x_loadings(
    structure,
    predictor_style="bar",
    predictor_names=feature_names,
    components=[0, 1],
)

y_loading_figure, y_loading_axes = plot_y_loadings(
    structure,
    response_names=target_names,
    components=[0, 1],
)

coefficient_figure, coefficient_axes = plot_coefficients(
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

The Pulp workflow described in [`examples.md`](examples.md#complete-pi-pls-analyses) demonstrates
the package inspection and plotting APIs together with example-owned I/O. It derives predictor and
response names visibly from the Pulp CSV headers, reads fixed component choices from the canonical
path artifacts, and clones the fixed Pi-PLS estimator
inside the same five non-shuffled folds, writes seven long-form CSV files under
`examples/results/pulp_post_analysis/`, rereads them, and constructs one seven-page PDF, including the Pulp biplot. The OOF
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
All four response headers (`TS`, `CP`, `ADF`, and `IVOMD`) are retained.

The Tobacco workflow described in [`examples.md`](examples.md#complete-pi-pls-analyses) preserves
the strictly decreasing wavenumber coordinate read from the `X.csv` headers and uses it for Pi-PLS
predictor directions, Pi-PLS X loadings, and response-specific coefficient curves. All
thirteen response headers are partitioned into deterministic source-order pages of at most five
responses; the report renderer rejects pagination that omits, duplicates, or reorders a response.
The Pulp report derives its biplot from the canonical `x_scores.csv` and `x_loadings.csv`; no additional biplot table is required.

Tobacco writes the seven common post-analysis tables plus
`observation_diagnostics.csv`, whose columns are `sample`, `score_distance`, and
`x_reconstruction_residual`. The report plots these raw values without theoretical outlier limits.
