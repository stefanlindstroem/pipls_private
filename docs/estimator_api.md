# `PiPLSRegression`

`PiPLSRegression` is the direct estimator for one fixed Pi-PLS model. It follows the
scikit-learn fit/predict interface and does not perform cross-validation or parameter selection.

```python
from pipls import PiPLSRegression

model = PiPLSRegression(
    n_components=2,
    predictor_rank=4,
).fit(X_train, Y_train)
Y_pred = model.predict(X_test)
```

Constructor parameters are `n_components`, `scale`, `copy`, `predictor_rank`, `svd_solver`, and
`random_state`. Both ranks are explicit positive integers and must satisfy
`n_components <= predictor_rank`.

Use `PiPLSPathCV` for the standard bounded triangular search over component count and predictor
rank. A fitted `PiPLSRegression` has no `cv_results_`, `best_params_`, OOF predictions, or other
selection attributes.

## Preprocessing

`scale=True` centers and divides predictor and response columns by training-sample standard
deviations with `ddof=1`. `scale=False` centers without division. Constant columns use scale 1.
Every fit owns its learned preprocessing statistics. During `PiPLSPathCV`, each fixed estimator
clone learns them only from its training fold.

## Statistical-support warning

A direct fixed fit emits `StatisticalSupportWarning` when

\[
\frac{n}{r_\pi}<4.
\]

This is a diagnostic, not a rank-selection rule. Algebraically infeasible ranks and ranks above the
verified numerical rank remain errors.

## PLS-style estimator surface

`predict(X, copy=True)`, `transform(X, y=None, copy=True)`, `fit_transform(X, y)`, and
`inverse_transform(X, y=None)` mirror the corresponding `PLSRegression` conventions.
`transform(X)` returns predictor scores; supplying `y` returns `(x_scores, y_scores)`.

Standard fitted attributes include `x_weights_`, `y_weights_`, `x_loadings_`, `y_loadings_`,
`x_scores_`, `y_scores_`, `x_rotations_`, `y_rotations_`, `coef_`, and `intercept_`. Because Pi-PLS
uses direct orthogonal score maps rather than iterative deflation, weights and rotations coincide.

Pi-PLS-specific factorization output is available through the public frozen
`PiPLSDecomposition` instance at `decomposition_`. It contains `Pi`, `C`, `W`, `P`, `D`, `Q`, the
dilation vector, numerical-rank diagnostics, and the resolved predictor SVD solver.

`response_scale_for_scoring_` is the safe training-response sample standard deviation used by the
package response-standardized scorer. `predictor_rank_` records the fitted explicit rank, while
`max_predictor_rank_` records the centered algebraic limit `min(n_features, n_samples - 1)` for the
supplied training data.
