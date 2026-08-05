# Numerical contracts

## Linear algebra and rank

- Use thin SVDs and avoid forming a predictor covariance matrix when `p >> n`.
- Use `numpy.linalg.eigh` only for explicitly symmetrized matrices.
- Do not form explicit inverses. Use solves, least squares, SVDs, or documented pseudoinverses.
- The fixed core classifies predictor singular values with
  `tau_X = max(n, p) * eps64 * s_1`; requested predictor rank above the verified numerical rank
  raises `ValueError`.
- Repeated or nearly repeated singular values identify subspaces, not stable signed basis columns.
  Compare projectors, principal angles, singular values, regression maps, or predictions.
- Invalid dimensions are errors and are never silently clamped.

## Estimator preprocessing and finite behavior

Every fixed fit learns predictor and response means from its own training observations. With
`scale=True`, safe sample standard deviations use `ddof=1`; with `scale=False`, both blocks are
still centered and unit scales are stored. Search candidates learn these quantities independently
inside each training fold, and `refit()` learns them again from the supplied full training data.

Public fits are transactional. A failed fit removes old and partial fitted state. A failed search
refit leaves the fitted search unchanged because refitting operates on a fresh clone.

Finite float64 inputs must either produce finite public fitted/output arrays or raise a clear
exception. Range-safe fallbacks are allowed only where ordinary finite calculations overflow or a
nonconstant column underflows to zero scale. Public fixed fitting rejects magnitudes that would
make the core predictor-response product unrepresentable.

`copy=False` may reuse independent writable arrays, but read-only inputs and overlapping predictor
and response storage must be copied when needed for correctness.

## Search feasibility and candidate evaluation

The search materializes one split set and verifies predictor feasibility in every transformed
training fold. With:

- `p_min`: minimum transformed feature count;
- `n_train_min`: minimum materialized training-fold size;
- `r_num_min`: minimum verified fold numerical rank;
- `n`: total observations supplied to `fit()`;
- `c = samples_per_predictor_rank`;

the rule-derived ceiling is:

```text
min(p_min, n_train_min - 1, r_num_min, ceil(n / c))
```

`c` is positive and finite; the ceiling operation is normative. Candidate pairs satisfy
`1 <= n_components <= min(n_targets, predictor_rank)` and are fixed-model clones. Fold-local
preprocessing, scoring, and warning suppression must not leak validation data.

Adaptive rank search is deterministic for fixed inputs and configuration. Snapshot 509 names the
exhaustive mode `search_method="optimal"`; Decision 0149 renames the two coverage values to
`"adaptive"` and `"exhaustive"` in Patch 2 without changing either algorithm. The fitted
`search_is_exhaustive_` diagnostic continues to report achieved coverage, so an adaptive request may
still report exhaustive coverage on a sufficiently small or fully refined candidate interval.
Private numerical tie behavior uses dedicated numerical tie tolerances and deterministic
smaller-rank ordering.

Decision 0148 preserves that private comparison for exact-reference identification,
`rank_test_score`, and adaptive refinement. Its accepted final retained-rank rule uses separate
public tolerances. For fixed component count $h$ and exact reference score $S_{h,\max}$, evaluated
rank $r$ qualifies when both

\begin{equation}
S_{hr}
\geq
S_{h,\max}
-
\delta_{\mathrm{rel},r}
\left|S_{h,\max}\right|
\end{equation}

and

\begin{equation}
S_{hr}
\geq
S_{h,\max}
-
\delta_{\mathrm{abs},r}.
\end{equation}

The smallest evaluated qualifying rank is retained. Adaptive candidate coverage remains determined
by the exact score optimum and must not depend on these public tolerances.

## Response-standardized MSE

For each response, residuals are divided by the sample standard deviation learned from the
candidate's training responses. The scalar loss is the uniform mean over observations and response
columns. Constant response columns use the package's safe training-scale contract.

Path CV-MSE statistics give every materialized validation split equal weight, including repeated CV
and unequal validation-set lengths:

```text
cv_mse_mean = np.mean(split_cv_mse)
cv_mse_std  = np.std(split_cv_mse, ddof=0)
```

A one-split path therefore has `cv_mse_std == 0`. `cv_mse_std` is descriptive population SD across
splits; no standard error is derived or exposed.

## Selection tolerances

Predictor-rank and component-count tolerances are separate stages. Predictor-rank tolerances are
configured on `PiPLSSearchCV` and act in configured-score units within each component count.
Component-count tolerances are supplied to `select()` or `refit()` and act on the resulting
conditioned path in CV-MSE units.

Let `M_min` be the exact minimum stored path mean. For a resolved nonnegative relative tolerance
`delta_rel` and nonnegative absolute tolerance `delta_abs`:

```text
T_rel = (1 + delta_rel) * M_min
T_abs = M_min + delta_abs
T     = min(T_rel, T_abs)
```

The selected row is the smallest stored component count with `cv_mse_mean <= T`. Both caps must
hold, exact boundary equality is eligible, and a zero minimum keeps the relative cap at zero.

`relative_tolerance=None` resolves to `sqrt(np.finfo(np.float64).eps)`. Relative tolerance must be
finite and nonnegative. Absolute tolerance may also be positive infinity, which disables that cap;
NaN, negative values, and negative infinity are invalid. Tolerances apply only to
`rule="minimum_cv_mse"`.

The component-count selection retains an unruled exact-minimum path row, resolved tolerances, and
a derived effective threshold. Under Decision 0148, that reference row retains its own independent
predictor-rank evidence. Direct component lookup and `best_score` have no component-count tolerance
provenance but still carry predictor-rank evidence for optimized policies. All provenance must remain
exactly reproducible for OOF compatibility validation.

## OOF reporting

`oof_report()` reuses defensive copies of every split materialized by the fitted search. Each
selected fixed pair is refitted on every training fold. Multiple validation predictions for one
observation are averaged. Counts record how many predictions contributed.

Coverage representation is exact:

- positive count: complete finite prediction row;
- zero count: complete NaN prediction row.

Pooled OOF R2 is computed only over covered observations and is `None` when fewer than two covered
rows make it undefined. The report does not rescore candidates or fit a full-data model.

## Immutable public results

Public result records defensively copy arrays, make them read-only, normalize accepted NumPy
scalars, validate direct construction, and revalidate during pickle reconstruction. Counts and
ranks are positive; `n_components <= predictor_rank`; score-like fields are finite; MSE and SD are
nonnegative; component-aligned arrays have exact compatible shapes.

Inspection helpers must return finite derived arrays or raise `ValueError`. Display-factor sign
changes must preserve `P D Q.T`; balanced biplot scaling must preserve the selected `T P.T`
reconstruction; prediction residuals use `observed - predicted` and response standardization uses
observed-response sample scales with `ddof=1`.

## Numerical review

Every numerical change must state its comparison tolerance and add at least one boundary-case test.
When behavior should not change, compare before/after arrays or scalar results directly rather than
relying only on the test suite.
