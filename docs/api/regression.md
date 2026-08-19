# Fixed Π-PLS regression

Use `PiPLSRegression` when `n_components` and `predictor_rank` are already fixed. Here
`n_components` is the number of paired latent modes $h$, while `predictor_rank` is the retained
predictor-subspace dimension $r_\pi$. Both are required keyword-only constructor arguments because
the estimator fits exactly one explicit pair and does not perform cross-validation or parameter
selection.

```python
from pipls import PiPLSRegression

model = PiPLSRegression(
    n_components=2,
    predictor_rank=4,
).fit(X_train, Y_train)

Y_pred = model.predict(X_test)
```

Equations use $\mathbf{Y}$ for the response matrix, while Python signatures use scikit-learn's
conventional `y` name even for multivariate responses. The
[API overview](index.md#mathematical-notation-and-python-names) records this notation boundary.

For the normal path-selection workflow, begin with the
[synthetic tutorial](../tutorials/synthetic.md). Use `PiPLSSearchCV` when the rank pair is not already
fixed. Common fit and data problems are summarized in
[Troubleshooting](../troubleshooting.md).

## Selection provenance on refitted models

A model returned by `PiPLSSearchCV.refit()` exposes the exact immutable component-path selection
that configured it:

```python
search = PiPLSSearchCV().fit(X, Y)
selection = search.select(rule="minimum_cv_mse")
model = search.refit(X, Y, selection=selection)
assert model.selection_ is selection
```

For a pipeline template, `selection_` belongs to the returned outer pipeline. The terminal
`PiPLSRegression` step remains an ordinary fixed-pair estimator. A `PiPLSRegression` fitted directly
through `fit()` has no `selection_` because no search selection occurred. In evidence-retaining
workflows, the pre-existing selection is the handoff to OOF reporting and refitting;
`model.selection_` confirms the provenance of the successful final fit.

## Response-subspace selection

`response_subspace` controls how the intermediate orthonormal response basis $\mathbf{C}$ is
selected before the shared least-squares coupling and final diagonalization. Exactly two values are
supported:

- `"cross_covariance"` maximizes retained predictor-response cross-covariance. This is the
  peer-reviewed Π-PLS construction and the package default.
- `"least_squares"` chooses the response subspace that minimizes the rank-$h$ training
  least-squares residual after the predictor subspace has been fixed. It is an
  RRR-inspired software extension and is **not part of the peer-reviewed companion publication**.

For a fixed pair $(h,r_\pi)$, request the software extension directly:

```python
model = PiPLSRegression(
    n_components=3,
    predictor_rank=9,
    response_subspace="least_squares",
).fit(X_train, Y_train)
```

Both policies use the same downstream least-squares estimate of the reduced coupling, followed by
the same diagonalization into predictor directions $\mathbf{P}$, dilation $\mathbf{D}$, and response
directions $\mathbf{Q}$. Thus `response_subspace` selects the intermediate basis $\mathbf{C}$; the
public fitted response directions remain $\mathbf{Q}$. See
[Response-subspace selection](../theory.md#response-subspace-selection) for the mathematical
criteria and their RRR relationship.
A complete matched-CV programming example is available in
[Example 07](../examples.md#compare-response-subspace-policies).

When manuscript alignment matters, use the default `"cross_covariance"` explicitly in recorded
configuration. See [Companion-manuscript synthetic data](../manuscript_reproduction.md).

## Preprocessing and fit safety

Every fit centers predictors and responses using statistics learned from that fit's training data.
`scale` remains the backward-compatible default for both blocks: with `scale=True`, centered
predictor and response columns are divided by sample standard deviations with `ddof=1`; with
`scale=False`, both blocks remain only centered. Constant columns use scale 1.

Predictor and response scaling can be overridden independently with `scale_x` and `scale_y`.
Each defaults to `None`, which inherits `scale`. For example, an estimator placed after an external
predictor-scaling transformer can preserve those transformed predictor units while retaining
ordinary response standardization:

```python
model = PiPLSRegression(
    n_components=2,
    predictor_rank=4,
    scale_x=False,
    scale_y=True,
)
```

Centering remains active for both blocks under every combination. This separation is useful for
scikit-learn pipelines that learn predictor preprocessing inside each training fold: the upstream
transformer owns predictor scaling, while the terminal Π-PLS estimator can still standardize the
responses. Explicit `scale_x` or `scale_y` values take precedence over `scale`.

Ordinary means and standard deviations are retained for ordinary data. Range-safe fallbacks are
used only when finite values would otherwise overflow or when a nonconstant scale would underflow
to zero. Fits are transactional: a failed fit removes partial state and any earlier fitted model.
Successful public fitted values, predictions, transformations, and inverse reconstructions must be
finite. With `copy=False`, independent writable arrays may be modified in place; read-only arrays
and overlapping predictor/response storage are copied where mutation would be unsafe.

## Solver and statistical support

`svd_solver` accepts `"full"`, `"randomized"`, or `"auto"`. Only the predictor SVD may be
randomized. `random_state` follows the ordinary scikit-learn forms; the default `0` makes randomized
SVD reproducible. Randomized predictor SVD can be useful when matrices are large and the retained
rank is well below their dimensions; it is an approximate route and is not necessarily faster near
full rank. See
[Computational performance](../computational_performance.md#use-randomized-predictor-svd-for-large-problems).

A direct fixed fit emits `PredictorRankSupportWarning` when $n/r_\pi<3$. The warning is diagnostic and
does not alter the requested rank. Algebraically or numerically infeasible ranks remain errors.
`PiPLSSearchCV` does not impose an EPV support ceiling on ordinary automatic search; EPV is an
explicit fixed-rank policy requested with `predictor_rank_values="epv"`.

::: pipls.PredictorRankSupportWarning
    options:
      members: false

## Fitted results

Standard PLS-family fitted attributes include scores, loadings, rotations, coefficients, and
intercepts. The frozen, read-only `decomposition_` result groups the Π-PLS factorization and
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

## Π-PLS decomposition

`PiPLSRegression.decomposition_` is normally obtained from a fitted estimator. Directly
constructed instances apply the same shape, finite-value, scalar, and read-only-array validation.
Its field names state the mathematical objects directly: the orthonormal predictor and response
directions defined in the
[canonical terminology](../theory.md#canonical-terminology):

| Field | Method notation | Meaning |
|---|---|---|
| `predictor_directions` | $\mathbf{P}$ | orthonormal predictor directions |
| `dilation` | $D_k=D_{kk}$ | nonnegative dilation of each paired latent mode |
| `response_directions` | $\mathbf{Q}$ | orthonormal response directions |
| `predictor_numerical_rank` | — | complete numerical rank under full SVD, or a verified lower bound under randomized SVD |
| `predictor_numerical_rank_is_exact` | — | whether the reported numerical rank is complete |
| `rank_tolerance` | — | tolerance used to classify retained predictor singular values |
| `predictor_svd_solver` | — | predictor SVD implementation actually used |
| `standardized_regression_map` | $\mathbf{P}\mathbf{D}\mathbf{Q}^{\mathsf T}$ | regression map in centered/scaled coordinates |

The rank fields distinguish the algebraic fixed-fit ceiling `max_predictor_rank_` from the
numerical rank verified by the fitted decomposition. With randomized predictor SVD, the reported
rank is a verified lower bound rather than a complete rank calculation. The estimator transforms
the centered/scaled map back to original predictor and response units when constructing `coef_`,
`intercept_`, and prediction output.

::: pipls.decomposition.PiPLSDecomposition
    options:
      show_signature: false
      members:
        - standardized_regression_map
