# Project map

## Purpose

Pi-PLS is a PLS-family method for multivariate regression. The repository provides a
scientifically explicit numerical core, scikit-learn-compatible public estimators, model-selection
and validation utilities, reproducible datasets, and paper-reproduction infrastructure.

For a fresh-chat handoff, read `.llm/state.md` before using this map. That file records the current
implemented boundary and next increment; this file records where responsibilities live.

## Current state

Phases A through E1 are implemented. The current public surface includes `PiPLSRegression`,
`PiPLSPathCV`, `PiPLSDecomposition`, `PiPLSValidationReport`, public selection metrics, and
`StatisticalSupportWarning`.

The next increment is Phase E2: the dataset registry and generic loader. No real dataset migration, implicit download, or paper reproduction belongs in E2.

## Runtime ownership

- `src/pipls/_core.py`: fixed-`(n_components, predictor_rank)` numerical core.
- `src/pipls/_cv_engine.py`: shared fold-local candidate evaluation, scoring, timing, caching, and
  OOF refitting support.
- `src/pipls/_sklearn_compat.py`: cross-version estimator-aware validation and tags.
- `src/pipls/decomposition.py`: immutable public Pi-PLS factorization result.
- `src/pipls/datasets.py`: immutable dataset container and deterministic synthetic generators.
- `src/pipls/exceptions.py`: package warning and exception types.
- `src/pipls/metrics.py`: response-standardized selection metrics.
- `src/pipls/model_selection.py`: rank limits, split materialization, and shared rank-search
  orchestration.
- `src/pipls/path.py`: pipeline-aware `PiPLSPathCV` meta-estimator.
- `src/pipls/regression.py`: `PiPLSRegression` fixed-model estimator and conditional rank search.
- `src/pipls/validation.py`: immutable validation and OOF reporting.
- `src/pipls/__init__.py`: deliberate top-level public exports.

## Test ownership

- `tests/unit/`: local behavior and boundary conditions.
- `tests/invariants/`: mathematical identities, dimensions, orthogonality, and subspace properties.
- `tests/integration/`: estimator composition, leakage boundaries, and shared-engine equivalence.
- `tests/api/`: exposed parameter validation, scikit-learn/PLS compatibility, and validation
  protocols.
- `tests/estimator_checks/`: applicable scikit-learn common estimator checks.
- `tests/regression/`: frozen comparisons with trusted implementations.
- `tests/test_repository_seed.py`: `.llm` navigation, snapshot layout, and workflow invariants.

## Dataset and reproduction ownership

- `datasets/registry.yaml`: E2 registry source; entries must be schema-validated before use.
- `datasets/`: dataset descriptions and committed redistributable artifacts only.
- `scripts/prepare_data/`: deterministic converters and download/verification orchestration.
- `scripts/reproduce_paper/`: explicit paper-specific analysis workflows.
- `examples/`: small executable API demonstrations, not publication pipelines.
- `paper/`: manuscript-facing metadata and reproduction entry points.

## Contract and documentation ownership

- `.llm/state.md`: current handoff, accepted scope, and revised roadmap.
- `.llm/strategy.md`: phase history, acceptance conditions, and maintenance protocol.
- `.llm/decisions.md`: navigation for accepted decision records.
- `.llm/theory.md`: persistent conceptual derivation based on the manuscript.
- `.llm/mathematics.md`: concise normative mathematical contract.
- `.llm/numerical_contracts.md`: numerical policy and degeneracy behavior.
- `.llm/public_api.md`: public constructors, methods, outputs, defaults, and exclusions.
- `.llm/development.md`: implementation, testing, patch, and documentation rules.
- `docs/decisions/`: accepted design records.
- `docs/publication_repository_plan.md`: broad historical architecture and publication rationale;
  current accepted decisions may narrow older proposals.

## Architectural invariants

- Runtime code does not import from `.llm`, tests, examples, docs, scripts, datasets, or paper.
- The fixed numerical core does not own preprocessing, CV, datasets, or paper policy.
- `PiPLSRegression` and `PiPLSPathCV` do not wrap each other; both use shared private machinery.
- Learned preprocessing is fitted inside the corresponding training fold.
- Public behavior changes include focused tests and contract/documentation updates.
- Generated files, archive clutter, and unverified datasets are not committed.
- Changes are small, testable, and returned as root-relative unified Git patches.

## Validation

```bash
make check
make build
```

Use `make build` whenever packaging, dependencies, public modules, or included data files change.
Record each validation target as passed, failed, or not run.
