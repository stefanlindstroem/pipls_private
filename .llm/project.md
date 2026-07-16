# Project map

## Purpose

Pi-PLS is a PLS-family method for multivariate regression. The publication repository will
provide a theory-faithful numerical core, a scikit-learn estimator API, model-selection
utilities, reproducible datasets, and paper-reproduction scripts.

## Current increment

Phase D1 is implemented. `PiPLSPathCV` evaluates the admissible triangular
`(n_components, predictor_rank)` surface, clones complete pipelines inside every fold, mirrors
the established `"optimal"` and `"auto"` search policies, and exposes refitted best-estimator
plus reconstructable path diagnostics.

The next increment is Phase D2: ordered out-of-fold predictions, Leave-One-Out reporting, and
advanced split protocols.

## Planned responsibilities

- `src/pipls/_core.py`: fixed-$(h,r_\pi)$ numerical core.
- `src/pipls/_validation.py`: common parameter and array validation.
- `src/pipls/regression.py`: `PiPLSRegression`.
- `src/pipls/model_selection.py`: rank limits and private CV-selection primitives.
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
