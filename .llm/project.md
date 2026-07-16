# Project map

## Purpose

Pi-PLS is a PLS-family method for multivariate regression. The publication repository will
provide a theory-faithful numerical core, a scikit-learn estimator API, model-selection
utilities, reproducible datasets, and paper-reproduction scripts.

## Current increment

Phase C1 adds the shared ceiling-based predictor-rank bound and the
`PiPLSRegression(predictor_rank="max")` rule-fixed mode. The fixed core and explicit-integer public
estimator are already implemented. Automatic rank selection, path analysis, dataset APIs, and
plotting remain pending.

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
