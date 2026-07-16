# Project map

## Purpose

Pi-PLS is a PLS-family method for multivariate regression. The publication repository will
provide a theory-faithful numerical core, a scikit-learn estimator API, model-selection
utilities, reproducible datasets, and paper-reproduction scripts.

## Current increment

Phase D1, the shared-selection refactor, and the public API-alignment increment are implemented.
`PiPLSRegression` now follows PLS-style scikit-learn method, feature-name, output-container, and
fitted-attribute conventions; `PiPLSPathCV` exposes the selected nested estimator and preserves
indexable containers inside pipeline folds. Both selection interfaces reuse one private
candidate-evaluation engine and one adaptive rank-search engine.

The next increment is Phase D2: ordered out-of-fold predictions, Leave-One-Out reporting, and
advanced split protocols.

## Planned responsibilities

- `src/pipls/_core.py`: fixed-$(h,r_\pi)$ numerical core.
- `src/pipls/_cv_engine.py`: shared fold-local candidate evaluation and caching.
- `src/pipls/_sklearn_compat.py`: cross-version estimator-aware validation.
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
