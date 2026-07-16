# Numerical contracts

- Use thin SVDs and avoid forming a $p\times p$ covariance matrix when $p\gg n$.
- Use `numpy.linalg.eigh` only for symmetric matrices, after explicit symmetrization.
- Do not form explicit inverses. Use solves, least squares, SVDs, or documented pseudoinverses.
- The fixed core uses $\tau_X=\max(n,p)\,\epsilon_{64}\,s_1$ for predictor numerical rank; requested predictor rank above that numerical rank raises `ValueError`.
- The shared rule-derived upper predictor rank is `min(p, n_train_min, ceil(n_train_min / c))`; `c` must be positive and finite, and the ceiling operation is normative.
- Constant columns, rank deficiency, $p\gg n$, and nearly repeated singular values require deterministic behavior.
- Singular/eigenvector signs are not identifiers.
- Basis equality is not required when only the spanned subspace is identifiable.
- All public fitted arrays must be finite unless an input validation error is raised.
- Invalid requested dimensions raise errors; they are not silently clamped.
- Numerical changes must include a boundary-case test and state the tolerance used.
