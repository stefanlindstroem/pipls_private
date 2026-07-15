# Development contract

- Follow scikit-learn estimator conventions for constructor parameters, cloning, validation, fitted attributes, and scalar `score()`.
- Keep constructor arguments unchanged; resolve data-dependent values in `fit()`.
- Add dependencies only when a short, stable NumPy/scikit-learn implementation is insufficient.
- Every behavioral change requires focused tests.
- Mathematical changes update `.llm/mathematics.md` and user-facing theory documentation when applicable.
- Numerical changes update `.llm/numerical_contracts.md` when policy changes.
- Do not edit generated files or commit caches, build products, archive clutter, or unverified datasets.
- Return unified Git patches relative to repository root.
- Report each validation target as passed, failed, or not run.
