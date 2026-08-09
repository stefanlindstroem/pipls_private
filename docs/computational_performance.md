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

The default adaptive method evaluates a deterministic subset of admissible predictor ranks:

```python
from pipls import PiPLSSearchCV

search = PiPLSSearchCV(
    search_method="adaptive",
    cv=development_cv,
).fit(X, Y)
```

`"adaptive"` normally reduces $N_{\mathrm{pair}}$, but it may omit admissible ranks. It is therefore
not guaranteed to identify the same exact score optimum as exhaustive evaluation for an arbitrary
CV-score curve.

Use exhaustive coverage when complete evaluation of the admissible ranks is required:

```python
reference_search = PiPLSSearchCV(
    search_method="exhaustive",
    cv=development_cv,
).fit(X, Y)
```

`"exhaustive"` is useful as a reference on smaller representative problems or when complete
candidate coverage is itself part of the analysis. `search.search_is_exhaustive_` reports achieved
coverage: a small adaptive search can still evaluate every admissible pair and return `True`.

Predictor-rank tolerances are applied after candidate evaluation. They can retain a smaller rank
from the evaluated candidates, but they do not reduce adaptive or exhaustive search work.

## Use fixed or restricted predictor-rank policies when justified

When the scientific model already specifies the predictor-rank policy, avoid scanning ranks that
will not be considered. This can reduce training computational cost significantly.

Use the largest supported predictor rank across the component path:

```python
search = PiPLSSearchCV(
    n_components_values="all",
    predictor_rank_values="max",
    samples_per_predictor_rank=10.0,
    cv=development_cv,
).fit(X, Y)
```

Here `predictor_rank_values="max"` does not search over predictor ranks. It fixes the retained
predictor rank at the resolved support ceiling and evaluates the requested component counts at that
rank. With the default rule-based ceiling, the resolved value is

\begin{equation}
r_{\pi,\mathrm{max}}
=
\min\left[
    p_{\mathrm{min}},
    n_{\mathrm{train,min}}-1,
    r_{\mathrm{num,min}},
    \left\lceil\frac{n}{c}\right\rceil
\right],
\end{equation}

where $c$ is `samples_per_predictor_rank`, $n$ is the number of observations supplied to `fit()`,
and the other terms enforce fold-wise predictor dimensions and verified numerical rank. When those
feasibility limits are inactive, this is the practical $r_\pi=\min(p,\lceil n/c\rceil)$ rule.

For the events-per-variable (EPV) heuristic policy, set $c$ explicitly:

```python
# Conservative default used for most real-data analyses.
search = PiPLSSearchCV(
    n_components_values="all",
    predictor_rank_values="max",
    samples_per_predictor_rank=10.0,
    cv=final_cv,
).fit(X, Y)

# More permissive small-sample variant.
small_sample_search = PiPLSSearchCV(
    n_components_values="all",
    predictor_rank_values="max",
    samples_per_predictor_rank=5.0,
    cv=final_cv,
).fit(X, Y)
```

Lowering $c$ permits a larger supported predictor subspace; increasing $c$ makes the rank ceiling
more conservative. With `n_components_values="all"`, the component path then runs from one through
the largest feasible paired-mode count at that fixed maximum predictor rank. This is different from
the default conditional predictor-rank search, which evaluates or selects predictor ranks separately
for each component count.

Use one declared fixed rank:

```python
search = PiPLSSearchCV(
    predictor_rank_values=[20],
    cv=development_cv,
).fit(X, Y)
```

Or restrict the search to a declared subset. Request exhaustive coverage when every supplied value
must be evaluated:

```python
search = PiPLSSearchCV(
    predictor_rank_values=[10, 20, 40, 80],
    search_method="exhaustive",
    cv=development_cv,
).fit(X, Y)
```

`"max"` and a one-element sequence provide one rank candidate per component count, so their
examples intentionally omit `search_method`. These configurations are not faster versions of
unrestricted rank selection: they define different model or candidate policies. Predictor-rank
tolerances do not apply to `"max"` or a one-element sequence.

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

Only the predictor SVD is randomized. The response-side and latent coupling decompositions remain
exact. An integer `random_state` makes the randomized route reproducible.

Randomized SVD is an approximate numerical route, not a universal acceleration. It is most useful
when the requested predictor rank is small compared with both predictor-matrix dimensions. Near
full rank, its advantage can disappear and `svd_solver="full"` can be competitive. When feasible,
compare `"randomized"` with `"full"` on a representative subset or on the final selected fixed fit.

For a very large problem where maximum retained predictor rank is also the intended model policy,
the controls can be combined without a rank-search method:

```python
search = PiPLSSearchCV(
    estimator=large_problem_template,
    predictor_rank_values="max",
    cv=development_cv,
    n_jobs=-1,
).fit(X, Y)
```

This combination is useful only when the resulting maximum feasible rank remains sufficiently below
the matrix dimensions for randomized SVD to be advantageous.

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

1. Begin with one seeded shuffled partition, adaptive rank coverage, and a measured worker count.
2. Confirm data handling, fold-local preprocessing, selection, OOF reporting, and final refitting.
3. If the search remains too costly, decide whether maximum or fixed predictor rank, a restricted
   rank set, or a restricted component path is scientifically justified.
4. For sufficiently large predictor matrices, compare randomized and full predictor SVD on a
   representative problem.
5. Restore the validation repetitions, candidate policy, SVD policy, and diagnostics required by
   the final analysis.
6. Record the final protocol together with the selected pair and achieved candidate coverage.

The development configuration is a workflow aid. Results intended for interpretation or reporting
must come from the final declared protocol, not from a temporarily narrowed search unless that
narrowed policy is itself the intended analysis.

## Summary of trade-offs

| Control | Main computational effect | What changes? |
|---|---|---|
| Fewer CV repetitions | Fewer fold-local fits | Validation and partition-sensitivity evidence |
| `"adaptive"` rather than `"exhaustive"` | Usually fewer evaluated ranks | Candidate coverage; the exact reference optimum can differ |
| `predictor_rank_values="max"` | One rank per component count | Predictor-rank model policy |
| Fixed or restricted rank values | Fewer rank candidates | Candidate set or fixed-rank policy |
| Restricted `n_components_values` | Fewer component candidates | Evaluated component path |
| `svd_solver="randomized"` | Potentially cheaper suitable predictor decompositions | Approximate predictor-SVD route |
| Larger `n_jobs` | May reduce wall time | Execution and memory use; no intended statistical change |
| Reuse one `oof_report()` | Avoids repeated selected-pair fold fits | No change to the fitted search or report |
