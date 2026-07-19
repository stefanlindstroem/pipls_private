# Numerical model inspection

`pipls.inspection` contains fitted-model analysis computations that do not depend on pandas or
Matplotlib. `pipls.plotting` consumes those immutable results and provides optional Matplotlib
figures. Dataset-specific tables, OOF orchestration, and multipage reports remain separate example
workflows.

Import these names from the submodule:

```python
from pipls.inspection import (
    PLSLatentStructure,
    PLSObservationDiagnostics,
    PiPLSDisplayFactors,
    PredictionDiagnostics,
    pipls_display_factors,
    pls_latent_structure,
    pls_observation_diagnostics,
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

## Ordinary PLS latent structure

`pls_latent_structure()` accepts a fitted scikit-learn `PLSRegression` model and copies only its
public fitted arrays:

```python
from sklearn.cross_decomposition import PLSRegression
from pipls.inspection import pls_latent_structure

pls_model = PLSRegression(n_components=2, scale=True).fit(X, Y)
pls_structure = pls_latent_structure(pls_model)
```

The immutable result contains:

- `x_scores`, with shape `(n_samples, n_components)`;
- `x_loadings`, with shape `(n_features, n_components)`;
- `y_loadings`, with shape `(n_targets, n_components)`;
- `coefficients`, with the public scikit-learn orientation `(n_targets, n_features)`.

The arrays are defensive read-only copies. No scores, loadings, or coefficients are recomputed,
rescaled, or sign-adjusted by `pipls`. These quantities describe the fitted ordinary PLS model and
are not validation results.

## Ordinary PLS observation diagnostics

`pls_observation_diagnostics()` accepts a fitted `PLSRegression` and explicit predictor observations:

```python
from pipls.inspection import pls_observation_diagnostics

observation_diagnostics = pls_observation_diagnostics(pls_model, X)
```

Let $t_i$ be the transformed score of supplied observation $i$, and let
$\bar t_{\mathrm{train}}$ and $S_{T,\mathrm{train}}$ be the center and sample covariance of
the fitted training scores. The squared score distance is

\begin{equation}
h_i=(t_i-\bar t_{\mathrm{train}})^\mathsf{T}S_{T,\mathrm{train}}^{+}(t_i-\bar t_{\mathrm{train}}).
\end{equation}

Here $+$ denotes the Moore--Penrose inverse. The row-wise squared X-reconstruction residual is

\begin{equation}
q_i=\lVert x_i-\hat x_i\rVert_2^2,
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

Those lists can then be passed to categorical Pi-PLS and ordinary PLS plots. This is an
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
    plot_pls_coefficients,
    plot_pls_observation_diagnostics,
    plot_pls_scores,
    plot_pls_x_loadings,
    plot_pls_y_loadings,
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

Raw ordinary PLS observation diagnostics use a separate result object:

```python
observation_figure, observation_axes = plot_pls_observation_diagnostics(
    observation_diagnostics,
)
```

The function returns one `observation_diagnostics` axis and adds neither theoretical limits nor
automatic observation labels.


### Ordinary PLS figures

The ordinary PLS functions consume `PLSLatentStructure`, not an estimator:

```python
score_figure, score_axes = plot_pls_scores(
    pls_structure,
    components=(0, 1),
)

x_loading_figure, x_loading_axes = plot_pls_x_loadings(
    pls_structure,
    predictor_style="bar",
    predictor_names=feature_names,
    components=[0, 1],
)

y_loading_figure, y_loading_axes = plot_pls_y_loadings(
    pls_structure,
    response_names=target_names,
    components=[0, 1],
)

coefficient_figure, coefficient_axes = plot_pls_coefficients(
    pls_structure,
    predictor_style="bar",
    predictor_names=feature_names,
    response_names=target_names,
    responses=[0, 2],
)
```

`plot_pls_scores()` requires exactly two distinct zero-based component indices. Optional sample
labels annotate observations, but the function does not infer groups or draw confidence regions.
X loadings and response-specific coefficients share the explicit bar-versus-line predictor
contract. Selected X-loading components use grouped bars or overlaid lines on one axis. Selected
Y-loading components use grouped bars on one response axis. Selected coefficient responses use
grouped bars or overlaid lines on one predictor axis. Categorical displays require caller-supplied
predictor or response names. For line rendering, the caller supplies the physical predictor
coordinate and axis label, whose order is preserved.

The ordinary PLS plotting surface also includes `plot_pls_observation_diagnostics()`, which draws
raw score distance against squared X-reconstruction residual. It does not include theoretical
limits, automatic observation labels, or contribution plots. Biplots, confidence ellipses, VIP,
automatic variable selection, uncertainty intervals, and permutation tests remain outside the
implemented surface.

The complete [`10_pulp_real_data.py`](../examples/10_pulp_real_data.py) workflow demonstrates the
package inspection and plotting APIs together with example-owned I/O. It derives predictor and
response names visibly from the Pulp CSV headers, reads fixed component choices from the canonical
path artifacts, and clones those fixed estimators
inside the same five non-shuffled folds, writes seven long-form CSV files under
`examples/results/pulp_post_analysis/`, rereads them, and constructs one seven-page PDF. The OOF
predictions are labeled `selection-conditioned OOF predictions` because the fixed parameters were
chosen after examining paths computed from the same observations.

The standardized response columns in `predictions.csv` use centers and sample standard deviations
computed from the complete supplied observed-response matrix for display. They are not the
fold-local response scales used by the component-path scorer, so their aggregate squared values
should not be expected to equal the reported mean fold CV-MSE exactly.

The complete [`11_sugarcane_real_data.py`](../examples/11_sugarcane_real_data.py) workflow uses the
same seven-table contract under `examples/results/sugarcane_post_analysis/`. The script reads the
physical wavelength coordinate separately from the numeric `X.csv` headers, verifies that it is
strictly increasing, and passes it to the report renderer with the label `Wavelength (nm)`. The
canonical loading, direction, and coefficient values remain in long-form CSV tables; the report
uses line rendering only because the example supplies the ordered physical coordinate explicitly.
All four response headers (`TS`, `CP`, `ADF`, and `IVOMD`) are retained.

The complete [`12_tobacco_real_data.py`](../examples/12_tobacco_real_data.py) workflow preserves
the strictly decreasing wavenumber coordinate read from the `X.csv` headers and uses it for Pi-PLS
predictor directions, ordinary PLS X loadings, and response-specific coefficient curves. All
thirteen response headers are partitioned into deterministic source-order pages of at most five
responses; the report renderer rejects pagination that omits, duplicates, or reorders a response.
Tobacco writes the seven common post-analysis tables plus
`pls_observation_diagnostics.csv`, whose columns are `sample`, `score_distance`, and
`x_reconstruction_residual`. The report plots these raw values without theoretical outlier limits.
