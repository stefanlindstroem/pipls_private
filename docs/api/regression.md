# Fixed Pi-PLS regression

Use `PiPLSRegression` when `n_components` and `predictor_rank` are already fixed. The estimator does
not perform cross-validation or parameter selection.

```python
from pipls import PiPLSRegression

model = PiPLSRegression(
    n_components=2,
    predictor_rank=4,
).fit(X_train, Y_train)

Y_pred = model.predict(X_test)
```

For the normal path-selection workflow, begin with the
[Pulp tutorial](../tutorials/pulp.md). Use `PiPLSPathCV` when the rank pair is not already fixed.

## Preprocessing and fit safety

Every fit centers predictors and responses using statistics learned from that fit's training data.
With `scale=True`, centered columns are divided by sample standard deviations with `ddof=1`;
constant columns use scale 1. With `scale=False`, centering remains active without division.

During `PiPLSPathCV`, every candidate clone learns these statistics inside its training fold. Any
additional learned preprocessing should therefore be placed in a supported scikit-learn pipeline,
not fitted on the complete dataset before path evaluation.

Ordinary means and standard deviations are retained for ordinary data. Range-safe fallbacks are
used only when finite values would otherwise overflow or when a nonconstant scale would underflow
to zero. Fits are transactional: a failed fit removes partial state and any earlier fitted model.
Successful public fitted values, predictions, transformations, and inverse reconstructions must be
finite. With `copy=False`, independent writable arrays may be modified in place; read-only arrays
and overlapping predictor/response storage are copied where mutation would be unsafe.

## Solver and statistical support

`svd_solver` accepts `"full"`, `"randomized"`, or `"auto"`. Only the predictor SVD may be
randomized. `random_state` follows the ordinary scikit-learn forms; the default `0` makes randomized
SVD reproducible.

A direct fixed fit emits `StatisticalSupportWarning` when $n/r_\pi<3$. The warning is diagnostic and
does not alter the requested rank. Algebraically or numerically infeasible ranks remain errors. The
standard path search uses its more conservative default support ceiling.

## Fitted results

Standard PLS-family fitted attributes include scores, weights, loadings, rotations, coefficients,
and intercepts. Pi-PLS-specific matrices and numerical-rank diagnostics are grouped in the frozen,
read-only `decomposition_` result. The exact attribute shapes and conditional method behavior are
documented below.

::: pipls.PiPLSRegression
    options:
      members:
        - fit
        - predict
        - transform
        - fit_transform
        - inverse_transform
        - score
        - get_feature_names_out
