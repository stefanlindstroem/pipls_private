# Project map

## Purpose

Pi-PLS is a PLS-family method for multivariate regression. The publication repository will
provide a theory-faithful numerical core, a scikit-learn estimator API, model-selection
utilities, reproducible datasets, and paper-reproduction scripts.

## Current increment

Phase D2 is implemented. Both public interfaces accept grouped and other ordinary scikit-learn
splitters, expose optional row-ordered OOF predictions with repeat counts and partial-coverage
markers, reject foldwise R2 for singleton validation folds, and attach an immutable validation
report that explicitly labels selection-conditioned estimates.

The next increment is Phase E: common-format datasets and the deterministic synthetic generator.

## Planned responsibilities

- `src/pipls/_core.py`: fixed-$(h,r_\pi)$ numerical core.
- `src/pipls/_cv_engine.py`: shared fold-local candidate evaluation and caching.
- `src/pipls/_sklearn_compat.py`: cross-version estimator-aware validation.
- `src/pipls/validation.py`: immutable CV and OOF reporting.
- `src/pipls/decomposition.py`: public immutable Pi-PLS factorization result.
- `src/pipls/regression.py`: `PiPLSRegression`.
- `src/pipls/model_selection.py`: rank limits and shared rank-search orchestration.
- `src/pipls/path.py`: pipeline-aware `PiPLSPathCV`.
- `src/pipls/metrics.py`: response-standardized selection metrics.
- `tests/unit/`: local behavior.
- `tests/invariants/`: mathematical identities and subspace properties.
- `tests/integration/`: estimator composition and leakage boundaries.
- `tests/regression/`: frozen comparisons with trusted implementations.
- `.llm/theory.md`: persistent conceptual reference derived from the Pi-PLS manuscript.
- `.llm/mathematics.md`: concise normative mathematical contract.
- `.llm/`: repository communication contracts and workflow scripts.

## Invariants

- Runtime code does not import from `.llm`, tests, examples, docs, scripts, datasets, or paper.
- Changes are small, testable, and root-relative.
- Public behavior changes include tests and documentation.
- Generated files and unverified datasets are not committed.

## Validation

```bash
make check
make build
```

- D1c completed: final scikit-learn cleanup, conditional path delegation, inverse reconstruction, canonical decomposition arrays, standard sentinels/timings, and minimum-version CI.
