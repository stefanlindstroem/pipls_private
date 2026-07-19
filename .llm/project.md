# Project map

## Purpose

Pi-PLS is a PLS-family method for multivariate regression. The repository is the long-lived home of
the installable `pipls` package: its numerical core, scikit-learn-compatible public interfaces,
validation utilities, synthetic generators, user documentation, concise examples, staged fitted-
model analysis tools, transparent reference datasets, lightweight validation benchmarks, tests,
packaging, and releases.

The repository is not the reproduction environment for any one paper. Read
`.llm/product_scope.md` for the normative product/publication boundary.

For a fresh-chat handoff, read `.llm/state.md` before using this map. That file records the current
implemented boundary and next increment; this file records where responsibilities live.

## Current state

Phases A through E4c are implemented. The current public surface includes `PiPLSRegression`,
`PiPLSPathCV`, `PiPLSDecomposition`, `PiPLSValidationReport`, public selection metrics,
`StatisticalSupportWarning`, deterministic synthetic dataset generation, pure numerical
`pipls.inspection`, and optional Pi-PLS and ordinary PLS figures under `pipls.plotting`.

The transparent real-dataset suite contains pulp, sugarcane, and tobacco. A licensing review of
the remaining companion-analysis candidates intentionally excluded Corn, the legacy Citrination
Steel table, SARCOS, and FRED-MD from this repository because the exact source materials do not
carry sufficiently clear redistribution rights. Public navigation now describes the installable
package, API, examples, datasets, validation, and releases.
The over-general synthetic manifest, universal result schema, and broad CI runner have been removed.
The benchmark layer implements all four focused synthetic questions as separate scripts. Pulp,
Sugarcane, and Tobacco are component-path examples rather than benchmark or test-suite executions.
`make examples` runs every numbered example explicitly, including these complete real-data analyses.
They write separate Pi-PLS and standard PLS (NIPALS) CSVs, call imported helpers to derive
comparison PDFs from both canonical tables, and then fit a separately chosen fixed Pi-PLS model.
No block-aware scaling API is designed or scheduled.

## Implemented estimator and selection boundary

Decisions 0039 and 0040 are fully implemented. `PiPLSRegression` owns one explicit fixed rank
pair and no cross-validation or selection results. `PiPLSPathCV` owns the complete triangular-
selection lifecycle and defaults to the explicit complete-component sentinel `"all"`. Obsolete
private selection machinery and duplicate fitted aliases have been removed; Pi-PLS-specific output
is canonical in `decomposition_`. Supported pipelines infer their unique terminal Pi-PLS step and
carry their own output-container configuration through cloning and refit.

## Accepted analysis ownership

Decision 0042 adds a staged fitted-model analysis surface without changing the estimator or path
engine. The implemented `pipls.inspection` submodule owns immutable Pi-PLS display factors,
standardized prediction diagnostics, and copies of public ordinary PLS scores, loadings, and
coefficients. The optional `pipls.plotting` submodule renders Pi-PLS decomposition, prediction, and
ordinary PLS score/loading/coefficient figures from those explicit results. Dataset-specific model
choices, OOF loops, pandas tables, CSV writing, and multipage reports remain in `examples/`.

The current component-path helpers remain example-local selection-diagnostic tools. Full-data
decomposition, score, loading, and coefficient plots are interpretive. Prediction diagnostics must
receive predictions explicitly and record their provenance. Read `.llm/analysis.md` before
implementing or reviewing this surface.

## Runtime ownership

- `src/pipls/_core.py`: fixed-`(n_components, predictor_rank)` numerical core.
- `src/pipls/_cv_engine.py`: path-owned fold-local candidate evaluation, scoring, timing, caching,
  warning filtering, and OOF support.
- `src/pipls/_sklearn_compat.py`: cross-version estimator-aware validation and tags.
- `src/pipls/decomposition.py`: immutable public Pi-PLS factorization result.
- `src/pipls/datasets.py`: optional immutable dataset container and deterministic synthetic
  generators; it is not required for user-supplied real data.
- `src/pipls/exceptions.py`: package warning and exception types.
- `src/pipls/metrics.py`: response-standardized selection metrics.
- `src/pipls/inspection.py`: pure immutable fitted-model inspection computations.
- `src/pipls/plotting.py`: optional Matplotlib figures for explicit inspection results.
- `src/pipls/model_selection.py`: path-owned rank limits, split materialization, and rank-search
  orchestration.
- `src/pipls/path.py`: pipeline-aware `PiPLSPathCV` meta-estimator.
- `src/pipls/regression.py`: direct fixed-model `PiPLSRegression` estimator.
- `src/pipls/validation.py`: immutable validation and OOF reporting.
- `src/pipls/__init__.py`: deliberate top-level public exports.

## Test ownership

- `tests/unit/`: local behavior and boundary conditions.
- `tests/invariants/`: mathematical identities, dimensions, orthogonality, and subspace properties.
- `tests/integration/`: estimator composition and leakage boundaries.
- `tests/api/`: exposed parameter validation, scikit-learn/PLS compatibility, and validation
  protocols.
- `tests/estimator_checks/`: applicable scikit-learn common estimator checks.
- `tests/regression/`: frozen comparisons with trusted implementations.
- `tests/benchmarks/`: focused synthetic benchmark behavior and output-contract tests; future frozen
  fixtures require separate review.
- `tests/examples/`: small-data helper, CSV, PDF, and workflow-structure contracts; complete real-data
  examples are user-run and are not executed by the default test suite.
- `tests/test_repository_seed.py`: `.llm` navigation, snapshot layout, and workflow invariants.

## Product-asset ownership

- `datasets/`: committed redistributable real datasets using the standard `X.csv`, `Y.csv`, and
  `metadata.yaml` layout with public-only provenance and an explicit source-level redistribution
  grant for the exact included material. No generic runtime registry is required.
- `examples/`: small executable API demonstrations with transparent data reading, not hidden loader
  utilities or publication pipelines. Real-data examples keep separate Pi-PLS and standard PLS
  (NIPALS) CSV paths and derive comparison PDFs from those tables. Post-analysis examples own
  fixed-model OOF loops, canonical tables, physical-axis semantics, pagination, and report
  composition; the Pulp workflow is the first complete implementation. Their pandas and Matplotlib
  requirements are grouped in the `examples` optional
  dependency extra.
- `benchmarks/`: four focused synthetic package-validation scripts, each with one minimal generated
  CSV. Real-data analyses are not duplicated here.
- `docs/`: user and developer documentation, API guidance, mathematical contracts, release notes,
  and accepted decision records.
- packaging and release configuration: installable distributions, compatibility policy, versioning,
  and release automation.

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
- `.llm/benchmarking.md`: normative benchmark questions, minimal outputs, interpretation boundaries,
  and implementation order.
- `.llm/analysis.md`: normative fitted-model interpretation, prediction-diagnostic, plotting, and
  analysis-artifact contracts.
- `.llm/testing.md`: durable testing boundary.
- `.llm/development.md`: implementation, testing, patch, and documentation rules.
- `docs/decisions/`: accepted design records.

## Architectural invariants

- Runtime code does not import from `.llm`, tests, examples, docs, scripts, datasets, or benchmarks.
- The fixed numerical core does not own preprocessing, CV, datasets, benchmark policy, or
  publication workflows.
- Real-data input remains user-owned: examples form `X` and `Y` explicitly without a required
  registry or generic loader.
- `PiPLSRegression` and `PiPLSPathCV` do not wrap each other; selection machinery is owned by the
  path interface.
- `PiPLSRegression` owns current centering and optional scaling: every candidate fit learns its
  statistics from the corresponding training fold, and the selected model refits them on all
  supplied training data.
- Future block-aware variants of that standardization are valid product scope but currently have no
  accepted API, names, schedule, or implementation plan. They must preserve the same fold-local and
  full-training-refit boundary.
- Public behavior changes include focused tests and contract/documentation updates.
- Paper-specific figures, complete comparison grids, and reporting workflows belong in downstream
  repositories that pin tagged `pipls` releases.
- Generated files, archive clutter, and unverified datasets are not committed.
- Changes are small, testable, and returned as root-relative unified Git patches.

## Validation

```bash
make check
make examples
make build
```

Use `make examples` whenever numbered examples or their generated application artifacts change.
Use `make build` whenever packaging, dependencies, public modules, or included data files change.
Record each validation target as passed, failed, or not run.
