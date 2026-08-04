# Troubleshooting

This page collects common public-API problems. It focuses on actions a programming user can take
without relying on private implementation details.

## I do not know the two ranks

Use `PiPLSSearchCV` rather than guessing both values. The search object evaluates admissible
`(n_components, predictor_rank)` pairs, where `n_components` counts paired latent modes and
`predictor_rank` is the retained predictor-subspace dimension. It retains one conditionally selected
predictor rank for each paired-mode count and exposes the concise result through `component_path_`.

```python
search = PiPLSSearchCV().fit(X, Y)
path = search.component_path_
model = search.refit(X, Y, n_components=2)
```

Inspect `path` before choosing the component count. The returned model already uses the
conditionally selected predictor rank stored for that row. Continue with the
[synthetic tutorial](tutorials/synthetic.md) or the [path-selection reference](api/path.md).

## I need to change scaling or the SVD solver during search

`PiPLSSearchCV` has no separate `scale` or `svd_solver` parameter. Configure these settings on the
`PiPLSRegression` template supplied to the search:

```python
from pipls import PiPLSRegression, PiPLSSearchCV

template = PiPLSRegression(
    n_components=1,
    predictor_rank=1,
    scale=True,
    svd_solver="full",
    random_state=0,
)
search = PiPLSSearchCV(estimator=template).fit(X, Y)
```

The search replaces only `n_components` and `predictor_rank`; it retains the other template
settings while cloning candidates. With `estimator=None`, the ordinary `PiPLSRegression` defaults
are used. See [Configure the candidate estimator](api/path.md#configure-the-candidate-estimator).

## A component count or predictor rank was not evaluated

Inspect the values that were actually evaluated before requesting a result:

```python
search.component_path_.n_components
search.predictor_rank_profile(h).predictor_rank
```

An explicit component request must have at least one admissible predictor rank. Candidate limits
also depend on sample count, transformed training-fold dimensions, the minimum numerical rank
verified across those folds, configured rank bounds, and the selected search policy. See
[Path-selection details](path_analysis.md).

## The path search has no admissible candidate

Reduce the requested component count or predictor-rank range, provide more observations, remove
redundant predictors, or review `max_predictor_rank` and `samples_per_predictor_rank`. Grouped,
temporal, and other specialized splitters can reduce the smallest training-fold size or the minimum
verified fold rank and therefore the feasible path.

The [cross-validation protocols](path_analysis.md#cross-validation-protocols-and-metadata)
section explains splitter-dependent feasibility and metadata such as `groups`.

## `predict()` is unavailable after path selection

`PiPLSSearchCV` is a path evaluator rather than a fitted prediction model. Fit and retain one
selected model explicitly:

```python
model = search.refit(X, Y, n_components=h)
Y_pred = model.predict(X_new)
```

Automatic choices use the same post-fit operation:

```python
model = search.refit(X, Y, rule="one_standard_error")
```

The accepted rules are `"best_score"`, `"minimum_cv_mse"`, and the temporary
`"one_standard_error"`. `minimum_cv_mse` additionally accepts `relative_tolerance` and
`absolute_tolerance`; nondefault tolerances are rejected for every other rule and for manual
component-count selection. The search object retains the path evidence but does not delegate
prediction or store the returned model. See the [path API](api/path.md).

## The CV-MSE minimum, best candidate, and selected model disagree

Candidate selection always maximizes `mean_test_score`. With the default scorer, this is equivalent
to minimizing mean response-standardized CV-MSE. With a custom scorer, the CV-MSE columns remain
diagnostics and need not identify the selected candidate.

Use `search.select(rule="best_score")` for the global configured-score optimum. Use
`search.select(rule="minimum_cv_mse")` or `search.select(rule="one_standard_error")` for
fitting-free recommendation inspection. A final model is returned directly by `refit()` and records
the exact fitted row as `model.selection_`; it is not stored on the search. See
[Scoring and the best evaluated pair](path_analysis.md#scoring-and-the-best-evaluated-pair).

## A grouped splitter reports missing metadata

Pass groups to the search object's `fit()` call:

```python
search.fit(X, Y, groups=sample_groups)
```

Do not pass split metadata to `PiPLSRegression.fit()`, which fits one fixed pair and performs no
cross-validation. See
[cross-validation protocols and metadata](path_analysis.md#cross-validation-protocols-and-metadata).

## Fitting rejects the data or leaves the estimator unfitted

Public fits reject nonfinite inputs, constant columns that make the requested model undefined,
infeasible ranks, and numerical results that cannot be represented as finite `float64` values. A
failed fit is transactional: partial fitted state and any earlier fitted model are removed.

Correct the data or parameters and call `fit()` again. The
[fixed-regression reference](api/regression.md) documents preprocessing, fit safety, solver choices,
and the statistical-support warning.

## Inspection rejects an extreme derived quantity

Inspection helpers accept finite inputs but reject a derived norm, squared residual, standardized
value, or factor product that cannot be represented as finite `float64`. Rescale the physical units
or inspect the preceding fitted and prediction arrays. The package does not return nonfinite
inspection records.

## `PredictorRankSupportWarning` appears

A direct fixed fit warns when the number of observations is small relative to the requested
predictor rank. The warning does not change the requested model. It indicates weak statistical
support, while algebraically or numerically infeasible ranks remain errors.

Use a lower predictor rank, provide more observations, or use `PiPLSSearchCV`, whose default support
rule is more conservative. See
[Solver and statistical support](api/regression.md#solver-and-statistical-support).

## `copy=False` changed an input array

`copy=False` permits in-place centering and scaling of independent writable arrays. Use the default
`copy=True` when input preservation matters. Read-only arrays and overlapping predictor/response
storage are copied internally when mutation would be unsafe.

## An example cannot import Matplotlib or `adjustText`

Numerical estimators and inspection results do not require plotting dependencies. Install the
example extra before running the maintained plotting workflows:

```bash
python -m pip install ".[examples]"
```

The examples render immutable inspection arrays with ordinary Matplotlib. Annotated biplots also
use `adjustText`; call `adjust_text()` only after the final axis labels, limits, aspect, and legend
have been configured. Automatic label placement is heuristic and may need application-specific
tuning for dense
diagrams. It does not change `BiplotCoordinates`; inspect or export those arrays directly when a
non-graphical workflow is preferable.
