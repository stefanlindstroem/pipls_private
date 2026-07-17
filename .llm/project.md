# Project map

## Purpose

Pi-PLS is a PLS-family method for multivariate regression. The repository is the long-lived home of
the installable `pipls` package: its numerical core, scikit-learn-compatible public interfaces,
validation utilities, synthetic generators, user documentation, concise examples, transparent
reference datasets, lightweight validation benchmarks, tests, packaging, and releases.

The repository is not the reproduction environment for any one paper. Read
`.llm/product_scope.md` for the normative product/publication boundary.

For a fresh-chat handoff, read `.llm/state.md` before using this map. That file records the current
implemented boundary and next increment; this file records where responsibilities live.

## Current state

Phases A through E3 are implemented. The current public surface includes `PiPLSRegression`,
`PiPLSPathCV`, `PiPLSDecomposition`, `PiPLSValidationReport`, public selection metrics,
`StatisticalSupportWarning`, and deterministic synthetic dataset generation.

The initial transparent real-dataset suite contains Linnerud, pulp, sugarcane, and tobacco. The next
implementation increment is a repository-product cleanup: remove paper-oriented placeholders and
rewrite public navigation around package users. Runtime behavior and preprocessing APIs are not
part of that cleanup.

## Runtime ownership

- `src/pipls/_core.py`: fixed-`(n_components, predictor_rank)` numerical core.
- `src/pipls/_cv_engine.py`: shared fold-local candidate evaluation, scoring, timing, caching, and
  OOF refitting support.
- `src/pipls/_sklearn_compat.py`: cross-version estimator-aware validation and tags.
- `src/pipls/decomposition.py`: immutable public Pi-PLS factorization result.
- `src/pipls/datasets.py`: optional immutable dataset container and deterministic synthetic
  generators; it is not required for user-supplied real data.
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

## Product-asset ownership

- `datasets/`: committed redistributable real datasets using the standard `X.csv`, `Y.csv`, and
  `metadata.yaml` layout with public-only provenance. No generic runtime registry is required.
- `examples/`: small executable API demonstrations with transparent data reading, not hidden loader
  utilities or publication pipelines.
- `benchmarks/`: future lightweight package-validation manifests, runners, and documented generated
  outputs. Synthetic validation is primary; real datasets provide representative smoke checks.
- `docs/`: user and developer documentation, API guidance, mathematical contracts, release notes,
  and accepted decision records.
- packaging and release configuration: installable distributions, compatibility policy, versioning,
  and release automation.

The current `paper/` and `scripts/reproduce_paper/` directories are transitional placeholders from
an earlier publication-oriented plan. Decision 0024 rejects them as future package responsibilities;
the next cleanup patch should remove them and adjust public navigation without changing runtime
behavior.

## Contract and documentation ownership

- `.llm/state.md`: current handoff, accepted scope, and roadmap.
- `.llm/product_scope.md`: package-product and publication-reproduction boundary.
- `.llm/strategy.md`: increment history, acceptance conditions, and maintenance protocol.
- `.llm/decisions.md`: navigation for accepted decision records.
- `.llm/theory.md`: persistent conceptual derivation and scientific interpretation.
- `.llm/mathematics.md`: concise normative mathematical contract.
- `.llm/numerical_contracts.md`: numerical policy and degeneracy behavior.
- `.llm/public_api.md`: public constructors, methods, outputs, defaults, and exclusions.
- `.llm/data_io.md`: transparent real-data reading and example policy.
- `.llm/dataset_layout.md`: normative committed-dataset file and metadata convention.
- `.llm/testing.md`: durable testing boundary.
- `.llm/development.md`: implementation, testing, patch, and documentation rules.
- `docs/decisions/`: accepted design records.

## Architectural invariants

- Runtime code does not import from `.llm`, tests, examples, docs, scripts, datasets, or benchmarks.
- The fixed numerical core does not own preprocessing, CV, datasets, benchmark policy, or
  publication workflows.
- Real-data input remains user-owned: examples form `X` and `Y` explicitly without a required
  registry or generic loader.
- `PiPLSRegression` and `PiPLSPathCV` do not wrap each other; both use shared private machinery.
- Learned preprocessing is fitted inside the corresponding training fold.
- Future standardization or block scaling is valid product scope but currently has no accepted API,
  names, schedule, or implementation plan.
- Public behavior changes include focused tests and contract/documentation updates.
- Paper-specific figures, complete comparison grids, and reporting workflows belong in downstream
  repositories that pin tagged `pipls` releases.
- Generated files, archive clutter, and unverified datasets are not committed.
- Changes are small, testable, and returned as root-relative unified Git patches.

## Validation

```bash
make check
make build
```

Use `make build` whenever packaging, dependencies, public modules, or included data files change.
Record each validation target as passed, failed, or not run.
