# Computational performance

Π-PLS path training repeats fixed-rank estimator fits over a cross-validation protocol. Runtime is
therefore governed by more than the dimensions of `X` and `Y`: the number of validation splits, the
number of evaluated `(n_components, predictor_rank)` pairs, the retained ranks, preprocessing, the
predictor SVD policy, and parallel execution all matter.

These controls are not interchangeable. Fewer validation repetitions change the selection
evidence. Narrower rank and component sets change the evaluated model-selection problem.
Randomized predictor SVD changes the numerical route. `n_jobs` primarily changes execution, and
reusing an existing OOF report avoids repeated work. Learned preprocessing must remain inside every
fit and every training fold.

## A cost model for path training

Let $N_{\mathrm{split}}$ be the number of materialized validation splits and let
$N_{\mathrm{pair}}$ be the number of evaluated `(n_components, predictor_rank)` pairs. Candidate
evaluation performs

\begin{equation}
N_{\mathrm{candidate\ fit}}
=
N_{\mathrm{pair}}N_{\mathrm{split}}
\end{equation}

fold-local estimator fits. Each fit includes the complete estimator or pipeline, so centering,
optional scaling, and any pipeline preprocessing are learned again from that training fold.

Before evaluating candidates, `PiPLSSearchCV` also checks predictor-rank feasibility separately on
each training split. A useful accounting approximation for the Π-PLS fits performed by
`search.fit()` is therefore

\begin{equation}
N_{\mathrm{PiPLS\ fit}}
\approx
N_{\mathrm{split}}\left(N_{\mathrm{pair}}+1\right).
\end{equation}

The extra $N_{\mathrm{split}}$ term represents the fold-level feasibility probes. A subsequent
`search.refit()` adds one full-data fit. A call to `search.oof_report()` adds another
$N_{\mathrm{split}}$ fits for the selected pair. Repeated OOF calls repeat those fits.

This is an accounting model, not a wall-clock formula. Matrix dimensions, retained ranks, pipeline
steps, memory traffic, SVD implementation, candidate-batch sizes, and worker scheduling can make
two fits differ substantially in cost.

## Develop with a smaller validation protocol

The number of folds and repetitions is usually the largest simple multiplier. During workflow
development, use one seeded shuffled partition or a small number of repetitions. Increase the
validation effort when the final analysis requires a more detailed assessment of partition
sensitivity.

```python
from sklearn.model_selection import KFold, RepeatedKFold

# Fast development protocol: five materialized splits.
development_cv = KFold(
    n_splits=5,
    shuffle=True,
    random_state=0,
)

# More expensive final protocol: fifty materialized splits.
final_cv = RepeatedKFold(
    n_splits=5,
    n_repeats=10,
    random_state=0,
)
```

For the same candidate set, moving from the final protocol above to the development protocol
reduces the candidate-fold fit count by a factor of ten. It does not guarantee a tenfold wall-time
reduction, and it is not statistically neutral. Fewer repetitions provide less information about
how the selected path changes with the partition. Repeated CV also remains part of model selection;
it is not independent post-selection validation.

## Choose predictor-rank coverage deliberately

The default search is exhaustive over the complete hard-feasible predictor-rank domain:

```python
from pipls import PiPLSSearchCV

reference_search = PiPLSSearchCV(
    cv=development_cv,
).fit(X, Y)
```

If the largest available component count is $H$ and the hard/user predictor-rank ceiling is $R$,
with all integer ranks available, the default triangular candidate count is

\begin{equation}
N_{\mathrm{pair}}
=
\sum_{h=1}^{H}(R-h+1)
=
H(R+1)-\frac{H(H+1)}{2}.
\end{equation}

This is the cost of obtaining complete rank coverage. `samples_per_predictor_rank` does not reduce
that automatic domain.

For large problems, opt into adaptive coverage explicitly:

```python
search = PiPLSSearchCV(
    search_method="adaptive",
    cv=development_cv,
).fit(X, Y)
```

`"adaptive"` uses the same hard-feasible rank domain but normally evaluates a deterministic subset,
reducing $N_{\mathrm{pair}}$. Because admissible interior ranks may be omitted, it is not guaranteed
to identify the same exact score optimum as exhaustive evaluation for an arbitrary CV-score curve.
`search.search_is_exhaustive_` reports achieved coverage: a small adaptive search can still evaluate
every admissible pair and return `True`.

Predictor-rank tolerances are applied after candidate evaluation. They can retain a smaller rank
from the evaluated candidates, but they do not reduce adaptive or exhaustive search work.

## Use EPV, fixed ranks, or restricted rank domains only when intended

Rank-domain restrictions change the model-selection problem. Use them when they express the intended
analysis, not merely as an undocumented acceleration of the ordinary automatic search.

### EPV policy

The manuscript-style events-per-variable-inspired policy fixes one predictor rank instead of
optimizing it. Request it explicitly:

```python
search = PiPLSSearchCV(
    predictor_rank_values="epv",
    samples_per_predictor_rank=10.0,
    cv=final_cv,
).fit(X, Y)
```

The nominal EPV rank is

\begin{equation}
r_{\pi,\mathrm{epv,nominal}}
=
\min\left[p,\left\lceil\frac{n}{c}\right\rceil\right],
\end{equation}

where $n$ is the full number of observations supplied to `fit()` and $c$ is
`samples_per_predictor_rank`. The effective rank is clipped only by fold-local dimensional and
numerical feasibility and by an explicit integer `max_predictor_rank`, if supplied.

The default EPV value is $c=10$. A more permissive small-sample choice is $c=5$:

```python
small_sample_search = PiPLSSearchCV(
    predictor_rank_values="epv",
    samples_per_predictor_rank=5.0,
    cv=final_cv,
).fit(X, Y)
```

Values below 5 are legal but emit `PredictorRankSupportWarning`. A user who deliberately wants the
most permissive EPV endpoint can therefore write:

```python
endpoint_search = PiPLSSearchCV(
    predictor_rank_values="epv",
    samples_per_predictor_rank=1.0,
    cv=development_cv,
).fit(X, Y)
```

With $c=1$, the nominal EPV rank is $\min(p,n)$; the effective rank is then determined by the hard
fold-feasibility ceiling. There is no separate `predictor_rank_values="max"` policy.

EPV is a fixed-rank policy, so predictor-rank tolerances do not apply. With
`n_components_values="all"`, the component path runs only through the largest component count
compatible with the effective EPV rank.

### Explicit fixed or restricted ranks

Use one declared fixed rank when it is part of the model definition:

```python
search = PiPLSSearchCV(
    predictor_rank_values=[20],
    cv=development_cv,
).fit(X, Y)
```

Or restrict optimization to a declared subset:

```python
search = PiPLSSearchCV(
    predictor_rank_values=[10, 20, 40, 80],
    cv=development_cv,
).fit(X, Y)
```

The constructor default `search_method="exhaustive"` evaluates every admissible supplied rank. Use
`search_method="adaptive"` only when approximate coverage of a multi-rank explicit domain is also
intended. A one-element sequence and EPV expose one rank per compatible component count and are
therefore exhaustively covered without a separate rank search.

An explicit positive integer `max_predictor_rank` is another user-imposed domain restriction. It
caps automatic, explicit-sequence, and EPV ranks after hard feasibility has been established.
`max_predictor_rank=None` leaves the domain uncapped by user policy.

## Restrict the component path when justified

If scientific knowledge gives a credible component-count range, declare it directly:

```python
search = PiPLSSearchCV(
    n_components_values=[1, 2, 3, 4, 5],
    cv=development_cv,
).fit(X, Y)
```

This can reduce both the number of component counts and the associated predictor-rank candidates.
It also removes the excluded component counts from the evidence. The restriction should therefore
be justified as part of the model definition, not described as a computationally equivalent
shortcut.

## Use randomized predictor SVD for large problems

For sufficiently large predictor matrices, an explicit randomized predictor SVD can reduce the cost
when the retained predictor rank is well below the matrix dimensions:

```python
from pipls import PiPLSRegression, PiPLSSearchCV

large_problem_template = PiPLSRegression(
    n_components=1,
    predictor_rank=1,
    svd_solver="randomized",
    random_state=0,
)

search = PiPLSSearchCV(
    estimator=large_problem_template,
    search_method="adaptive",
    cv=development_cv,
).fit(X, Y)
```

Only the predictor SVD is randomized. Response-subspace construction remains exact under either
policy: `"cross_covariance"` uses an exact SVD of $\mathbf{Z}^{\mathsf T}\mathbf{Y}$, while
`"least_squares"` uses exact reduced QR of $\mathbf{Z}$ followed by an exact SVD of
$\mathbf{Q}_{Z}^{\mathsf T}\mathbf{Y}$. The final coupling SVD also remains exact. An integer
`random_state` makes the randomized predictor route reproducible.

Randomized SVD is an approximate numerical route, not a universal acceleration. It is most useful
when the requested predictor rank is small compared with both predictor-matrix dimensions. Near
full rank, its advantage can disappear and `svd_solver="full"` can be competitive. When feasible,
compare `"randomized"` with `"full"` on a representative subset or on the final selected fixed fit.

For a very large problem where an EPV-fixed predictor rank is the intended model policy, the
controls can be combined directly:

```python
search = PiPLSSearchCV(
    estimator=large_problem_template,
    predictor_rank_values="epv",
    samples_per_predictor_rank=10.0,
    cv=development_cv,
    n_jobs=-1,
).fit(X, Y)
```

This combination is useful only when the resulting EPV rank remains sufficiently below the matrix
dimensions for randomized SVD to be advantageous. If the intended analysis instead optimizes
predictor rank, retain `predictor_rank_values=None` and use adaptive coverage or an explicit rank
range to control search cost.

## Treat response-subspace policy as a model choice

`response_subspace` is not a search-cost shortcut and `PiPLSSearchCV` does not optimize it
automatically. The default `"cross_covariance"` route forms
$\mathbf{Z}^{\mathsf T}\mathbf{Y}$ and computes its exact response-side SVD. The optional
`"least_squares"` route instead computes an exact reduced QR factorization of $\mathbf{Z}$ and an
exact SVD of $\mathbf{Q}_{Z}^{\mathsf T}\mathbf{Y}$ before using the same downstream coupling
and diagonalization. Its extra QR work means it should not be described as a general computational
acceleration.

The least-squares criterion is optimal for the rank-$h$ training least-squares problem after the
predictor subspace has been fixed, and it can be useful for some datasets. Whether that translates
into lower validation or test error is problem-dependent. Compare the policies through two separate
searches using the same materialized CV splits and otherwise matched configuration:

```python
from pipls import PiPLSRegression, PiPLSSearchCV

shared_search = dict(cv=splits, search_method="adaptive")

cross_covariance = PiPLSSearchCV(
    estimator=PiPLSRegression(
        n_components=1,
        predictor_rank=1,
        response_subspace="cross_covariance",
    ),
    **shared_search,
).fit(X, Y)

least_squares = PiPLSSearchCV(
    estimator=PiPLSRegression(
        n_components=1,
        predictor_rank=1,
        response_subspace="least_squares",
    ),
    **shared_search,
).fit(X, Y)
```

Here `splits` is the same materialized split sequence for both searches. The least-squares policy
is a software extension and is not part of the peer-reviewed companion publication.

## Use parallelism deliberately

`n_jobs` controls joblib parallelism:

```python
search = PiPLSSearchCV(
    search_method="adaptive",
    cv=development_cv,
    n_jobs=-1,
).fit(X, Y)
```

During path search, candidate pairs within one evaluation batch are parallel tasks. The validation
splits for one candidate are evaluated serially inside that task. Adaptive batches can contain only
a few candidates, so all workers are not necessarily occupied throughout the search.

`oof_report()` has a different boundary: it parallelizes the selected-pair fits across validation
splits. In both cases, more workers can increase peak memory because several estimator fits and
matrix operations are active concurrently. Measure `n_jobs=1`, a modest positive value, and
`n_jobs=-1` on the actual matrices rather than assuming that all available workers are fastest.
Changing `n_jobs` does not change the requested candidates or validation splits and has no intended
statistical effect.

## Avoid repeated OOF computation

OOF diagnostics are computed after selection and are not cached by the search. Store and reuse the
immutable report:

```python
selection = search.select(
    rule="minimum_cv_mse",
    relative_tolerance=0.10,
)
report = search.oof_report(X, Y, selection=selection)

# Reuse report in every table and figure that needs the same diagnostics.
```

Each call fits the selected pair on every stored split. Recomputing the same report for separate
figures or tables repeats that entire set of fits. OOF diagnostics describe the selected pair under
the stored search protocol; they are not an independent estimate from an untouched outer test set.

## Inspect the work performed

The fitted search reports the actual candidate and split counts:

```python
n_pairs = search.cv_results_["n_components"].size
n_splits = search.n_splits_

print(f"Evaluated pairs: {n_pairs}")
print(f"Validation splits: {n_splits}")
print(f"Candidate-fold fits: {n_pairs * n_splits}")
print(f"Fold-level feasibility probes: {n_splits}")
print(f"Exhaustive coverage: {search.search_is_exhaustive_}")
```

`cv_results_` also contains per-candidate summaries across validation splits:

```python
mean_fit_time = search.cv_results_["mean_fit_time"]
std_fit_time = search.cv_results_["std_fit_time"]
mean_score_time = search.cv_results_["mean_score_time"]
std_score_time = search.cv_results_["std_score_time"]
```

These arrays help locate expensive candidates and separate fitting from scoring. They are not total
search wall times, particularly when candidates are evaluated in parallel. They also exclude the
fold-level feasibility probes, explicit OOF reporting, and final `refit()`.

## Separate development and final-analysis workflows

A practical sequence is:

1. Begin with one seeded shuffled partition and a measured worker count. On a large rank domain,
   explicitly use adaptive coverage while developing the workflow.
2. Confirm data handling, fold-local preprocessing, selection, OOF reporting, and final refitting.
3. If the search remains too costly, decide whether adaptive coverage, an EPV or fixed-rank policy,
   a restricted rank set, or a restricted component path is scientifically justified.
4. For sufficiently large predictor matrices, compare randomized and full predictor SVD on a
   representative problem.
5. Restore the validation repetitions, rank-domain policy, coverage strategy, SVD policy, and
   diagnostics required by the final analysis.
6. Record the final protocol together with the selected pair and achieved candidate coverage.

The development configuration is a workflow aid. Results intended for interpretation or reporting
must come from the final declared protocol, not from a temporarily narrowed search unless that
narrowed policy is itself the intended analysis.

## Summary of trade-offs

| Control | Main computational effect | What changes? |
|---|---|---|
| Fewer CV repetitions | Fewer fold-local fits | Validation and partition-sensitivity evidence |
| `"adaptive"` rather than default `"exhaustive"` | Usually fewer evaluated ranks | Candidate coverage; the exact reference optimum can differ |
| `predictor_rank_values="epv"` | One rank per component count | Predictor-rank model policy; $c$ sets the EPV rank |
| Fixed or restricted rank values | Fewer rank candidates | Candidate set or fixed-rank policy |
| Integer `max_predictor_rank` | Truncates the admissible rank domain | User-declared search/model domain |
| Restricted `n_components_values` | Fewer component candidates | Evaluated component path |
| `svd_solver="randomized"` | Potentially cheaper suitable predictor decompositions | Approximate predictor-SVD route |
| `response_subspace="least_squares"` | Adds exact QR plus an exact response-side SVD | Response-subspace model policy; compare validation evidence separately |
| Larger `n_jobs` | May reduce wall time | Execution and memory use; no intended statistical change |
| Reuse one `oof_report()` | Avoids repeated selected-pair fold fits | No change to the fitted search or report |
