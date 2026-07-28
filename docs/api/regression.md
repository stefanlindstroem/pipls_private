# Fixed Pi-PLS regression

Use `PiPLSRegression` when `n_components` and `predictor_rank` are already fixed. Both are
required keyword-only constructor arguments because the estimator fits exactly one explicit pair
and does not perform cross-validation or parameter selection.

```python
from pipls import PiPLSRegression

model = PiPLSRegression(
    n_components=2,
    predictor_rank=4,
).fit(X_train, Y_train)

Y_pred = model.predict(X_test)
```

Equations use $Y$ for the response matrix, while Python signatures use scikit-learn's conventional
`y` name even for multivariate responses. The
[API overview](index.md#mathematical-notation-and-python-names) records this notation boundary.

For the normal path-selection workflow, begin with the
[synthetic tutorial](../tutorials/synthetic.md). Use `PiPLSSearchCV` when the rank pair is not already
fixed. Common fit and data problems are summarized in
[Troubleshooting](../troubleshooting.md).

## Preprocessing and fit safety

Every fit centers predictors and responses using statistics learned from that fit's training data.
With `scale=True`, centered columns are divided by sample standard deviations with `ddof=1`;
constant columns use scale 1. With `scale=False`, centering remains active without division.

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

::: pipls.StatisticalSupportWarning
    options:
      members: false

## Fitted results

Standard PLS-family fitted attributes include scores, loadings, rotations, coefficients, and
intercepts. The frozen, read-only `decomposition_` result groups the Pi-PLS factorization and
numerical-rank diagnostics. The exact attribute shapes and conditional method behavior are
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
        - set_output

## Pi-PLS decomposition

`PiPLSRegression.decomposition_` is normally obtained from a fitted estimator. Directly
constructed instances apply the same shape, finite-value, scalar, and read-only-array validation.
Its public fields use descriptive Python names:

| Field | Method notation | Meaning |
|---|---|---|
| `predictor_rotations` | $P$ | orthogonal predictor directions |
| `dilation` | $\operatorname{diag}(D)$ | nonnegative strength of each paired mode |
| `response_rotations` | $Q$ | orthogonal response directions |
| `predictor_numerical_rank` | — | complete numerical rank under full SVD, or a verified lower bound under randomized SVD |
| `predictor_numerical_rank_is_exact` | — | whether the reported numerical rank is complete |
| `rank_tolerance` | — | tolerance used to classify retained predictor singular values |
| `predictor_svd_solver` | — | predictor SVD implementation actually used |
| `standardized_regression_map` | $PDQ^{\mathsf T}$ | regression map in centered/scaled coordinates |

The rank fields distinguish the algebraic fixed-fit ceiling `max_predictor_rank_` from the
numerical rank verified by the fitted decomposition. With randomized predictor SVD, the reported
rank is a verified lower bound rather than a complete rank calculation. The estimator transforms
the centered/scaled map back to original predictor and response units when constructing `coef_`,
`intercept_`, and prediction output.

::: pipls.PiPLSDecomposition
    options:
      show_signature: false
      members:
        - standardized_regression_map
