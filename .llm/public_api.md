# Public API contract

## Current top-level API

```python
from pipls import PiPLSPathCV, PiPLSRegression
```

`PiPLSRegression` selects predictor rank for one fixed component count. `PiPLSPathCV` searches
the admissible two-parameter surface and is the required route for arbitrary learned
preprocessing inside the CV boundary. Public scoring
callables are available from `pipls.metrics`:

```python
from pipls.metrics import (
    neg_response_standardized_mean_squared_error,
    response_standardized_mean_squared_error,
)
```

## Public names

| Public name | Mathematical notation |
|---|---|
| `n_components` | $h$ |
| `predictor_rank` | $r_\pi$ |
| `samples_per_predictor_rank` | $c$ |
| `predictor_rank_` | fitted $r_\pi$ |
| `max_predictor_rank_` | rule-derived upper bound |

Do not expose constructor aliases named `h`, `r_pi`, or `c`.


## Public validation and warning contract

Validation occurs at the start of `fit()` before copying arrays or invoking NumPy, joblib, or
scikit-learn internals.

- `n_components` and integer `predictor_rank` are positive Python or NumPy integers; booleans and
  integral-valued floats are invalid.
- Integer `cv` is at least 2; splitter objects and explicit split iterables remain valid.
- `n_jobs` is `None` or a nonzero integer.
- `random_state` lies in $[0, 2^{32}-1]$. It may be `None` only with
  `svd_solver="full"`; `svd_solver="auto"` may randomize and therefore requires a seed.
- `scale` and `copy` are Python or NumPy booleans.

`StatisticalSupportWarning` is public from `pipls`. A rule-based fit emits it once when
`samples_per_predictor_rank < 5`, because the resulting upper rank bound may not have sufficient
statistical support to be trusted without external validation. Explicit integer ranks do not emit
this warning because they bypass the $c$-based bound.

## Predictor-rank modes

The estimator supports four public modes:

```python
PiPLSRegression(n_components=2, predictor_rank=4)
PiPLSRegression(n_components=2, predictor_rank="max")
PiPLSRegression(n_components=2, predictor_rank="optimal")
PiPLSRegression(n_components=2, predictor_rank="auto")
```

The default is `predictor_rank="auto"`.

- A positive integer fixes the predictor rank directly.
- `"max"` uses the rule-derived upper bound without CV search.
- `"optimal"` exhaustively evaluates every admissible integer rank and returns the CV optimum for
  the fixed split set, scorer, and low-rank tie rule.
- `"auto"` performs deterministic logarithmic coarse-to-fine exploration, caches all evaluated
  candidates, refines the interval around the best observed rank, and exhaustively finishes when
  the remaining interval contains at most 10 ranks. It is approximate for arbitrary non-unimodal
  CV curves.

Both CV modes materialize one split set, use fold-local preprocessing, select the largest
scikit-learn score with deterministic low-rank tie-breaking, and refit the selected fixed-rank
model on all data passed to `fit()`.

## Predictor SVD policy

`svd_solver` accepts `"full"`, `"randomized"`, or `"auto"`; the default is `"auto"`.
`random_state=0` makes randomized decomposition reproducible. Only the first SVD of the
centered/scaled predictor matrix may be randomized. The response-subspace and coupling SVDs remain
exact.

The automatic solver selects randomized SVD only when all of the following hold:

```python
min(n_samples, n_features) >= 500
n_samples * n_features >= 1_000_000
predictor_rank <= 0.2 * min(n_samples, n_features)
```

Explicit `"full"` and `"randomized"` choices override this rule. Internal-CV candidate fits reuse
the same solver policy and seed.

The upper bound is

\begin{equation}
r_{\pi,\max}
=
\min\left[
p,
n_{\mathrm{train,min}},
\left\lceil
\frac{n_{\mathrm{train,min}}}{\texttt{samples\_per\_predictor\_rank}}
\right\rceil
\right].
\end{equation}

For `"auto"` and `"optimal"`, `n_train_min` is derived from the materialized internal-CV splits. For `"max"`,
the samples supplied to `fit()` are the training data. An explicit integer bypasses the
rule-derived bound but remains subject to the core numerical-rank and dimensional checks.

## Internal search parameters

- `cv` accepts an integer, a scikit-learn splitter, or an iterable of `(train, validation)` pairs.
  The default `cv=5` is deterministic, unshuffled regression K-fold splitting through
  scikit-learn's `check_cv` behavior.
- `scoring` accepts a scikit-learn scorer name or scorer callable. The default is
  `"neg_response_standardized_mean_squared_error"`.
- `n_jobs` controls parallel evaluation across predictor-rank candidates through joblib.

The positive response-standardized utility follows the scorer signature `(estimator, X, y)`.
The negative version is suitable for scikit-learn search APIs, where larger scores are better.

## Fitted estimator behavior

The estimator provides `fit`, `predict`, `transform`, and scalar `score`. It exposes
`predictor_rank_`, `max_predictor_rank_`, preprocessing statistics, Pi-PLS factorization arrays,
latent scores, `coef_` in scikit-learn orientation, `coef_matrix_` in manuscript orientation, and
`intercept_`.

Every fitted estimator exposes `svd_solver_`, the predictor solver actually used, and
`x_rank_is_exact_`. Under full SVD, `x_rank_` is the complete numerical rank; under randomized
truncated SVD it is a verified lower bound for the retained subspace.

Cross-validated modes additionally expose:

- `predictor_rank_values_`;
- `predictor_rank_cv_svd_solvers_`, mapping each evaluated rank to its fold-level solvers;
- `predictor_rank_cv_results_` with split and mean scores plus response-standardized MSE;
- `best_score_`;
- `best_response_standardized_mse_` for the selected rank;
- `n_splits_`;
- `cv_n_train_min_`;
- `predictor_rank_evaluation_order_`;
- `predictor_rank_search_history_`;
- `predictor_rank_search_method_`;
- `predictor_rank_search_interval_`;
- `n_predictor_rank_candidates_`, `n_predictor_rank_evaluated_`, and
  `n_predictor_rank_skipped_`;
- `predictor_rank_search_exhaustive_`.

`response_scale_for_scoring_` is estimated from the data used to fit each estimator with
`ddof=1`, independently of whether `scale` is true or false.


## Path-analysis API

`PiPLSPathCV` defaults to exhaustive `search_method="optimal"`. Adaptive
`search_method="auto"` applies coarse-to-fine predictor-rank search independently for each
`n_components` value. The admissible grid satisfies

\[
1 \le h \le \min(q,r_{\pi,\max}), \qquad h \le r_\pi \le r_{\pi,\max}.
\]

The class accepts a direct estimator or a composite estimator containing one
`PiPLSRegression`. It clones and fits the complete estimator inside every fold and candidate. A
unique nested Pi-PLS step is inferred; deeper composites use `pipls_param_prefix`.

An explicit integer `max_predictor_rank` bypasses the samples-per-rank rule but remains capped
by the smallest fold-safe algebraic dimension. The default `"rule"` mode uses the smallest
training-fold size and the smallest predictor dimension reaching the Pi-PLS step.

Public fitted attributes include standard search attributes (`cv_results_`, `best_params_`,
`best_score_`, `best_estimator_`) plus conditional-path, surface, candidate-count, and search-
method diagnostics documented in `docs/path_analysis.md`.
