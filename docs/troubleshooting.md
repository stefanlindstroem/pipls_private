# Troubleshooting

This page collects common public-API problems. It focuses on actions a programming user can take
without relying on private implementation details.

## Do I need to choose `predictor_rank`?

Usually not. For ordinary use, treat `n_components` as the model-complexity parameter, as you would
in PLS. `PiPLSSearchCV` evaluates the component path and resolves one predictor rank conditionally
for every evaluated component count. See
[Interpretation of the two rank controls](theory.md#interpretation-of-the-ranks) for why $h$ and
$r_\pi$ play different roles.

```python
search = PiPLSSearchCV().fit(X, Y)
path = search.component_path_
model = search.refit(X, Y, n_components=2)
```

Inspect `path` before choosing the component count. The returned model already uses the retained
predictor rank stored for that row. Advanced users can inspect
`search.predictor_rank_profile(h)` or constrain the predictor-rank search when statistical support
or scientific interpretation motivates direct control. Continue with the
[synthetic tutorial](tutorials/synthetic.md) or the [path-selection reference](api/path.md).

## I need to change scaling, response-subspace selection, or the SVD solver during search

`PiPLSSearchCV` has no separate `scale`, `response_subspace`, or `svd_solver` parameter. Configure
these settings on the `PiPLSRegression` template supplied to the search:

```python
from pipls import PiPLSRegression, PiPLSSearchCV

template = PiPLSRegression(
    n_components=1,
    predictor_rank=1,
    response_subspace="least_squares",
    scale=True,
    svd_solver="full",
    random_state=0,
)
search = PiPLSSearchCV(estimator=template).fit(X, Y)
```

The search replaces only `n_components` and `predictor_rank`; it retains the other template
settings while cloning candidates. The example above therefore keeps `"least_squares"` fixed for
every candidate. With `estimator=None`, the ordinary `PiPLSRegression` defaults are used, including
`response_subspace="cross_covariance"`. See
[Configure the candidate estimator](api/path.md#configure-the-candidate-estimator).

## I want least-squares response-subspace selection during search

Supply a `PiPLSRegression` template with `response_subspace="least_squares"`, as in the preceding
example. The search does not compare response-subspace policies automatically. If the scientific
question is whether `"least_squares"` or `"cross_covariance"` predicts better for a dataset, run two
searches with otherwise matched configuration and the same materialized validation splits. The
least-squares route is a software extension outside the
[peer-reviewed companion publication](citation.md#companion-paper); use `"cross_covariance"` for
manuscript-aligned fitting.

See [Response-subspace selection](theory.md#response-subspace-selection) for the mathematical
difference and [Compare response-subspace policies](examples.md#compare-response-subspace-policies)
for the maintained matched-split comparison.

## A component count or predictor rank was not evaluated

Inspect the values that were actually evaluated before requesting a result:

```python
search.component_path_.n_components
search.predictor_rank_profile(h).predictor_rank
```

An explicit component request must have at least one admissible predictor rank. Candidate limits
also depend on sample count, transformed training-fold dimensions, the minimum numerical rank
verified across those folds, configured rank bounds, and the selected search policy. See the
[search-domain contract](selection_validation.md#search-bounds).

## The path search is too slow or uses too much memory

Start by inspecting the number of materialized validation splits and evaluated candidate pairs:

```python
n_pairs = search.cv_results_["n_components"].size
n_splits = search.n_splits_
print(n_pairs, n_splits, n_pairs * n_splits)
```

During development, use one seeded shuffled partition or fewer repetitions. Then decide whether
explicit adaptive rather than default exhaustive rank coverage, an EPV or fixed predictor-rank
policy, a restricted rank or component set, or randomized predictor SVD is justified by the
analysis. Use `n_jobs` only after measuring memory and wall time on the actual matrices, and retain
one `oof_report()` result rather than recomputing it for each output.

Fewer repetitions and narrower candidate policies change the evidence or model-selection question;
randomized SVD changes the numerical route. The fit-count accounting and statistical consequences
of these controls are summarized under
[Computational consequences](selection_validation.md#computational-consequences).

## The path search has no admissible candidate

Reduce the requested component count or predictor-rank range, provide more observations, remove
redundant predictors, or review an explicit integer `max_predictor_rank`. If
`predictor_rank_values="epv"` is active, also review `samples_per_predictor_rank`, because it defines
the nominal fixed EPV rank. Grouped, temporal, and other specialized splitters can reduce the
smallest training-fold size or the minimum verified fold rank and therefore the feasible path.

The [cross-validation protocol](selection_validation.md#cross-validation-protocols-and-metadata)
contract explains splitter-dependent feasibility and metadata such as `groups`.

## `predict()` is unavailable after path selection

`PiPLSSearchCV` is a path evaluator rather than a fitted prediction model. Fit and retain one
selected model explicitly:

```python
model = search.refit(X, Y, n_components=h)
Y_pred = model.predict(X_new)
```

Automatic choices use the same post-search operation:

```python
model = search.refit(X, Y, rule="minimum_cv_mse")
```

The accepted rules are `"best_score"` and `"minimum_cv_mse"`. `minimum_cv_mse` additionally accepts
`relative_tolerance` and `absolute_tolerance`; nondefault tolerances are rejected for `best_score`
and for manual component-count selection. The search object retains the path evidence but does not delegate
prediction or store the returned model. See the [path API](api/path.md).

## The CV-MSE minimum, best candidate, and selected model disagree

Candidate evaluation always orients `mean_test_score` so that larger values are better. Predictor-
rank tolerance may deliberately retain a smaller rank whose score is below the exact conditional
optimum, and component-count tolerance may then retain a smaller component count above the exact
conditioned-path CV-MSE minimum. With a custom scorer, the CV-MSE columns remain diagnostics and
need not identify either score-based reference.

Use `search.select(rule="best_score")` for the configured-score optimum on the predictor-rank-
conditioned component path. Use `search.select(rule="minimum_cv_mse")` for fitting-free inspection
of the component-count tolerance rule. Inspect `search.predictor_rank_profile(h)` to compare the
exact conditional rank optimum with the tolerance-retained rank at one component count. Pass the
same compatible selection to `oof_report()` and `refit()` when those operations must describe one
exact row. The returned model records that row as `model.selection_`; it is not stored on the search.
See
[Candidate coverage and scoring](selection_validation.md#scoring-and-conditioned-path-selection).

## A grouped splitter reports missing metadata

Pass groups to the search object's `fit()` call:

```python
search.fit(X, Y, groups=sample_groups)
```

Do not pass split metadata to `PiPLSRegression.fit()`, which fits one fixed pair and performs no
cross-validation. See
[cross-validation protocols and metadata](selection_validation.md#cross-validation-protocols-and-metadata).

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

## An example cannot import Matplotlib

Numerical estimators and inspection results do not require plotting dependencies. Install the
example extra before running the maintained plotting workflows:

```bash
python -m pip install ".[examples]"
```

The examples render immutable inspection arrays with ordinary Matplotlib. The annotated Pulp biplot
uses optional `textalloc` after the axis has been configured. When available, it places predictor
labels while avoiding other predictor labels and the predictor-arrow shafts; sample-score points are
not treated as obstacles. If `textalloc` is unavailable, labels remain at their original Matplotlib
endpoint positions. Automatic label placement is heuristic and does not change `BiplotCoordinates`;
inspect or export those arrays directly when a non-graphical workflow is preferable.
