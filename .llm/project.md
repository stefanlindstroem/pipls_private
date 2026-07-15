# Project map

## Purpose

Pi-PLS is a PLS-family method for multivariate regression. The publication repository will
provide a theory-faithful numerical core, a scikit-learn estimator API, model-selection
utilities, reproducible datasets, and paper-reproduction scripts.

## Current increment

This increment contains repository infrastructure only. There is no estimator yet.
Do not invent a provisional implementation in unrelated changes.

## Planned responsibilities

- `src/pipls/_core.py`: fixed-$(h,r_\pi)$ numerical core.
- `src/pipls/_validation.py`: common parameter and array validation.
- `src/pipls/regression.py`: `PiPLSRegression`.
- `src/pipls/model_selection.py`: rank limits and `PiPLSPathCV`.
- `src/pipls/metrics.py`: response-standardized selection metrics.
- `tests/unit/`: local behavior.
- `tests/invariants/`: mathematical identities and subspace properties.
- `tests/integration/`: estimator composition and leakage boundaries.
- `tests/regression/`: frozen comparisons with trusted implementations.
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
