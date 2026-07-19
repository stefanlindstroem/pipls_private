# Numerical model inspection

`pipls.inspection` contains fitted-model analysis computations that do not depend on pandas or
Matplotlib. The current surface covers deterministic display copies of the Pi-PLS factorization and
standardized prediction diagnostics. Plotting and dataset-specific report generation are separate
later stages.

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

The forthcoming `pipls.plotting` submodule will consume these immutable numerical results. pandas
tables, fixed-model OOF orchestration, CSV artifacts, and dataset-specific reports remain owned by
the examples.
