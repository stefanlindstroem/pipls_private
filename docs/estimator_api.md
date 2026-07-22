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
selection attributes. Fixed estimators remain compatible with ordinary scikit-learn meta-estimators
when every candidate supplies an explicit admissible rank pair, but the package examples recommend
`PiPLSPathCV` rather than a hand-built triangular grid.

## Preprocessing

`scale=True` centers and divides predictor and response columns by training-sample standard
deviations with `ddof=1`. `scale=False` centers without division. Constant columns use scale 1.
Every fit owns its learned preprocessing statistics. During `PiPLSPathCV`, each fixed estimator
clone learns them only from its training fold.

Ordinary means and sample standard deviations are retained for ordinary data. Range-safe fallbacks
are used only when finite values would otherwise overflow mean or scale calculation, or when a
nonconstant column would underflow to a zero scale. With `copy=False`, independent writable arrays
may be centered and scaled in place; read-only arrays and overlapping `X`/`y` storage are copied as
needed.

## Statistical-support warning

A direct fixed fit emits `StatisticalSupportWarning` when

\[
\frac{n}{r_\pi}<3.
\]

The warning is diagnostic and does not alter the requested rank. Algebraically or numerically
infeasible ranks remain errors. The standard path search uses a more conservative default support
ceiling with five supplied observations per retained predictor direction.

## Predictor SVD and random state

`svd_solver` accepts `"full"`, `"randomized"`, or `"auto"`. Only the predictor SVD may be
randomized; the response-side and final coupling SVDs remain exact.

`random_state` accepts the conventional scikit-learn forms:

- an integer in `[0, 2**32 - 1]` for repeatable randomized SVD;
- a NumPy `RandomState` instance;
- `None`, which uses NumPy's global random state and is not promised to be repeatable.

The default `random_state=0` keeps the default estimator reproducible when randomized SVD is used.
The solver actually used and numerical-rank diagnostics are recorded in `decomposition_`.

## Methods and fitted output

`predict(X, copy=True)`, `transform(X, y=None, copy=True)`, `fit_transform(X, y)`, and
`inverse_transform(X, y=None)` mirror the corresponding `PLSRegression` conventions.
`transform(X)` returns predictor scores; supplying `y` returns `(x_scores, y_scores)`.

Standard fitted attributes include `x_weights_`, `y_weights_`, `x_loadings_`, `y_loadings_`,
`x_scores_`, `y_scores_`, `x_rotations_`, `y_rotations_`, `coef_`, and `intercept_`. Because Pi-PLS
uses direct orthogonal score maps rather than iterative deflation, weights and rotations coincide.

Pi-PLS-specific factorization output is available through the public frozen
`PiPLSDecomposition` instance at `decomposition_`. It contains `Pi`, `C`, `W`, `P`, `D`, `Q`, the
dilation vector, numerical-rank diagnostics, and the resolved predictor SVD solver. Its arrays are
read-only. These values are not duplicated as top-level symbolic aliases; this keeps one canonical
location for method-specific internals while standard PLS-style attributes remain directly
available.

`response_scale_for_scoring_` is the safe training-response sample standard deviation used by the
package response-standardized scorer. `predictor_rank_` records the fitted explicit rank, while
`max_predictor_rank_` records the centered algebraic limit `min(n_features, n_samples - 1)` for the
supplied training data.

Fits are transactional. If fitting fails, partial fitted attributes and any earlier fitted model are
removed. Successful fits, predictions, transformations, and inverse reconstructions must be finite;
values outside float64 range raise a clear numerical exception instead of being returned as `NaN`
or infinity.


## Generated reference

The generated [fixed-regression reference](api/regression.md) gives the complete constructor
signature, fitted-attribute shapes, and method return contracts. The generated
[path-selection reference](api/path.md) documents the search results and which attributes and
methods depend on `refit=True` or `return_oof_predictions=True`.
