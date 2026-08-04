# Numerical contracts

- Use thin SVDs and avoid forming a $p\times p$ covariance matrix when $p\gg n$.
- Use `numpy.linalg.eigh` only for symmetric matrices, after explicit symmetrization.
- Do not form explicit inverses. Use solves, least squares, SVDs, or documented pseudoinverses.
- The fixed core uses $\tau_{\mathrm{X}}=\max(n,p)\,\epsilon_{64}\,s_1$ for predictor
  numerical rank; requested predictor rank above that numerical rank raises `ValueError`.
- The search meta-estimator verifies predictor rank separately in every transformed training fold before
  candidate evaluation. With `r_num_min` denoting the minimum verified fold rank, the shared
  rule-derived upper predictor rank is
  `min(p_min, n_train_min - 1, r_num_min, ceil(n / c))`, where `n` is the total number of
  observations supplied to `fit()`, `p_min` is the minimum transformed feature count, and
  `n_train_min` is the smallest materialized training-fold size. `c` must be positive and finite,
  and the ceiling operation is normative.
- Constant columns, rank deficiency, $p\gg n$, and nearly repeated singular values require deterministic behavior.
- Singular/eigenvector signs are not identifiers.
- Basis equality is not required when only the spanned subspace is identifiable.
- All public fitted arrays must be finite unless an input validation error is raised.
- Public fixed and path fits are transactional: any failed fit removes previous and partial fitted
  state, so scikit-learn fitted-state checks report the estimator as unfitted.
- Ordinary means and sample standard deviations remain the default calculations. Range-safe
  fallbacks apply only when finite data overflow those calculations or a nonconstant column
  underflows to a zero sample scale.
- Public fixed fitting rejects magnitudes that can overflow the core predictor-response
  cross-product. Prediction, transformation, inverse reconstruction, and response-standardized MSE
  must either return finite float64 values or raise a clear exception.
- Invalid requested dimensions raise errors; they are not silently clamped.
- Numerical changes must include a boundary-case test and state the tolerance used.
- Core public result records defensively copy arrays, make them read-only, normalize accepted
  NumPy scalars, and validate direct construction and pickle reconstruction. Component/rank/split
  counts are positive, `n_components <= predictor_rank`, score-like scalars and arrays are finite,
  response-standardized MSE and fold-SD values are nonnegative, and decomposition factors are
  finite and component-aligned.
- Validation-report OOF rows use one explicit coverage representation: positive counts require
  finite predictions, while zero counts require NaN predictions across the complete response row.
  Counts are nonnegative integer arrays and align one-to-one with prediction rows.
- Inspection result records defensively copy and freeze arrays, validate direct construction, and
  reconstruct through the same validation path when unpickled. Shape relationships, provenance,
  nonnegative diagnostics, positive display scales, factor weighting, and prediction-standardization
  relationships are enforced.
- Inspection helpers use range-safe scaled means, sample scales, norms, covariance products,
  squared residual norms, and RMSE calculations. Finite inputs must either produce finite float64
  inspection quantities or raise a clear `ValueError` naming the unrepresentable derived quantity.
- Score-distance covariance scaling uses one common scalar for fitted and supplied centered scores,
  preserving the Moore--Penrose quadratic form. Biplot balancing uses scaled column norms and a
  quotient of square roots. These safeguards do not alter ordinary finite results.

- Automatic-selection response scales are estimated from the matching training fold with `ddof=1`; zero scales and singleton-training-fold scales are replaced by 1.0.
- Response-standardized MSE uniformly averages squared residuals over validation samples and response columns after division by the matching fold-local response scales.
- Candidate score ties use `numpy.isclose` with `rtol=1e-12` and `atol=1e-15`. Global
  and conditional selection compare every candidate directly with the relevant maximum score, then
  choose the lexicographically smallest tied complexity. `rank_test_score` uses minimum ranks and
  anchors each tolerant group to its leading score; adjacent near-ties must not chain candidates
  that are not tied with the same group reference. The derived
  `PiPLSPredictorRankProfile.selection` result applies the same reference-anchored
  comparison and chooses the first tied row because profile ranks are strictly ascending.
- Component-path and predictor-rank-profile records may represent a valid one-split protocol. Their
  stored `cv_mse_std` is the population standard deviation across realized split MSE values,
  while the derived `cv_mse_standard_error` is `cv_mse_std / sqrt(n_splits - 1)`, equivalently
  the sample split standard deviation divided by `sqrt(n_splits)`. Derived arrays are finite
  read-only `float64` and are a conventional CV heuristic rather than confidence intervals. The
  property raises explicitly when fewer than two split values make the estimate undefined.
  Maintained CV-MSE figures use this derived quantity for symmetric $\pm 1$ standard-error bars;
  they do not use the stored split SD as `yerr`.
- Search-owned minimum-CV-MSE selection identifies the first exact `np.argmin(cv_mse_mean)` row as
  an unruled reference, derives simultaneous relative and absolute thresholds, and returns the first
  ascending path row at or below their minimum. `relative_tolerance=None` resolves to square root of
  float64 epsilon; positive-infinity absolute tolerance disables the absolute cap. The temporary
  `"one_standard_error"` rule uses the same exact reference row plus its derived standard error.
- CV splits are materialized once, validated, copied, and reused for rank preflight and every
  candidate. The samples-per-rank term uses total `n`; the smallest centered training fold supplies
  the dimensional cap `n_train_min - 1`, and the minimum verified fold rank supplies the numerical
  cap. Explicit requested ranks above the resolved ceiling fail before candidate scoring.
- Fold-rank preflight preserves NumPy global random state, uses the configured terminal estimator
  scaling and predictor-SVD policy, and suppresses only `PredictorRankSupportWarning`.

- Predictor SVD policy is independent of rank-search policy. `"full"` is the exact reference,
  `"randomized"` is explicit approximation, and `"auto"` randomizes only when
  `min(n, p) >= 500`, `n * p >= 1_000_000`, and
  `predictor_rank <= 0.2 * min(n, p)`.
- Only the predictor-matrix SVD may be randomized; the response cross-product and coupling SVDs
  remain exact.
- Randomized SVD requires a nonnegative integer seed. With randomized truncated SVD, `x_rank` is
  a verified retained-rank lower bound rather than the complete numerical rank; diagnostics must
  expose this distinction.

## Accepted CV-MSE tolerance transition target

Decision 0146 replaces the current fold-SE selection contract through a staged migration. The final
contract gives every materialized validation split equal weight:

```python
cv_mse_mean = np.mean(split_cv_mse)
cv_mse_std = np.std(split_cv_mse, ddof=0)
```

For path minimum `minimum`, resolved relative tolerance `rtol`, and resolved absolute tolerance
`atol`, the effective threshold is:

```python
min((1.0 + rtol) * minimum, minimum + atol)
```

The first ascending component-path row at or below that threshold is selected. The default relative
tolerance resolves to `sqrt(np.finfo(np.float64).eps)` and the default absolute tolerance is
positive infinity. Relative tolerance is finite and nonnegative; absolute tolerance is nonnegative
and may be positive infinity. Both restrictions apply simultaneously.

`cv_mse_std` is descriptive split variability and is not divided by a split count. Maintained plots
will use it directly for symmetric error bars. Until Patches 2–7 are applied, the preceding
split-SD/derived-SE and one-standard-error paragraphs describe the current implementation.
