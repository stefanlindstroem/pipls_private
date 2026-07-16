# Project map

## Purpose

Pi-PLS is a PLS-family method for multivariate regression. The publication repository will
provide a theory-faithful numerical core, a scikit-learn estimator API, model-selection
utilities, reproducible datasets, and paper-reproduction scripts.

## Current increment

Phase C2b is implemented: the estimator can exhaustively evaluate the fold-safe predictor-rank
grid with fold-local preprocessing, reusable splits, standard scorer orientation, diagnostics,
and full-data refitting. That exhaustive behavior is still exposed under the provisional name
`predictor_rank="auto"` in the current source.

The accepted next increment is Phase C2c. It will rename exhaustive search to
`predictor_rank="optimal"` and reserve `predictor_rank="auto"` for deterministic adaptive
coarse-to-fine rank search. Randomized SVD is a later, separate numerical-policy increment.

## Planned responsibilities

- `src/pipls/_core.py`: fixed-$(h,r_\pi)$ numerical core.
- `src/pipls/_validation.py`: common parameter and array validation.
- `src/pipls/regression.py`: `PiPLSRegression`.
- `src/pipls/model_selection.py`: rank limits, private CV-selection primitives, and later `PiPLSPathCV`.
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
