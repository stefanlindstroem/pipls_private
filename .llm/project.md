# Project map

## Purpose

Pi-PLS is a PLS-family method for multivariate regression. The publication repository will
provide a theory-faithful numerical core, a scikit-learn estimator API, model-selection
utilities, reproducible datasets, and paper-reproduction scripts.

## Current increment

Phase C2c is implemented. `predictor_rank="optimal"` exhaustively evaluates the admissible rank
set, while the default `predictor_rank="auto"` performs deterministic logarithmic coarse-to-fine
search with cached candidates, a final exhaustive local interval, and explicit search
diagnostics.

The next increment is Phase C2d: an explicit scalable linear-algebra policy with full,
randomized, and automatic SVD solver choices. Predictor-rank search approximation and
linear-algebra approximation remain separate contracts.

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
