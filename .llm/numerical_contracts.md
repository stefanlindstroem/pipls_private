# Numerical contracts

- Use thin SVDs and avoid forming a $p\times p$ covariance matrix when $p\gg n$.
- Use `numpy.linalg.eigh` only for symmetric matrices, after explicit symmetrization.
- Do not form explicit inverses. Use solves, least squares, SVDs, or documented pseudoinverses.
- The fixed core uses $\tau_X=\max(n,p)\,\epsilon_{64}\,s_1$ for predictor numerical rank; requested predictor rank above that numerical rank raises `ValueError`.
- The shared rule-derived upper predictor rank is `min(p, n_train_min - 1, ceil(n / c))`, where
  `n` is the total number of observations supplied to `fit()` and `n_train_min` is the smallest
  materialized training-fold size. `c` must be positive and finite, and the ceiling operation is
  normative.
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

- Automatic-selection response scales are estimated from the matching training fold with `ddof=1`; zero scales and singleton-training-fold scales are replaced by 1.0.
- Response-standardized MSE uniformly averages squared residuals over validation samples and response columns after division by the matching fold-local response scales.
- Conditional predictor-rank ties use `numpy.isclose` with `rtol=1e-12` and `atol=1e-15`, then choose the smallest tied predictor rank.
- CV splits are materialized once, validated, copied, and reused for every candidate. The
  samples-per-rank term uses total `n`; the smallest centered training fold supplies only the
  feasibility cap `n_train_min - 1`.

- Predictor SVD policy is independent of rank-search policy. `"full"` is the exact reference,
  `"randomized"` is explicit approximation, and `"auto"` randomizes only when
  `min(n, p) >= 500`, `n * p >= 1_000_000`, and
  `predictor_rank <= 0.2 * min(n, p)`.
- Only the predictor-matrix SVD may be randomized; the response cross-product and coupling SVDs
  remain exact.
- Randomized SVD requires a nonnegative integer seed. With randomized truncated SVD, `x_rank` is
  a verified retained-rank lower bound rather than the complete numerical rank; diagnostics must
  expose this distinction.
