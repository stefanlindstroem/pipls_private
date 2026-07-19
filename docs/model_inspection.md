# Numerical model inspection

`pipls.inspection` contains fitted-model analysis computations that do not depend on pandas or
Matplotlib. `pipls.plotting` consumes those immutable results and provides optional Matplotlib
figures. Dataset-specific tables, OOF orchestration, and multipage reports remain separate example
workflows.

Import these names from the submodule:

```python
from pipls.inspection import (
    PiPLSDisplayFactors,
    PredictionDiagnostics,
    pipls_display_factors,
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

## Interpretation boundary

Display factors describe a fixed fitted model. They are not validation results. Prediction
statistics mean different things depending on their provenance. In particular,
`selection-conditioned OOF predictions` indicate that the same observations contributed to an
earlier parameter-selection path, so they are not an independent post-selection performance
estimate.

## Plotting computed results

Install the optional plotting dependency with:

```bash
python -m pip install "pipls[plot]"
```

The plotting names remain in their own submodule:

```python
from pipls.plotting import plot_pipls_decomposition, plot_prediction_diagnostics
```

`plot_pipls_decomposition()` displays one small multiple per requested zero-based component index.
The first row contains predictor directions $P_{:k}$, the second contains weighted response
directions $d_kq_{:k}$, and the final axis contains the corresponding dilation values. Use bars for
a small scalar predictor set:

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

The supplied coordinate order is preserved, including decreasing wavenumber axes. The function does
not smooth, interpolate, reorder, or infer a spectral representation.

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

The fast [`09_model_inspection.py`](../examples/09_model_inspection.py) example fits a fixed Pi-PLS
model to deterministic synthetic training data, diagnoses predictions on a separate synthetic test
set, and writes two compact PDFs. pandas tables, fixed-model OOF orchestration, canonical CSV
artifacts, and dataset-specific multipage reports remain owned by later real-data example stages.
