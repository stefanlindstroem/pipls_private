# Project map

## Purpose

Pi-PLS is a PLS-family method for multivariate regression. The repository is the long-lived home of
the installable `pipls` package: its numerical core, scikit-learn-compatible public interfaces,
validation utilities, synthetic generators, user documentation, numbered examples, staged fitted-
model analysis tools, transparent reference datasets, lightweight validation benchmarks, tests,
packaging, and releases.

The repository is not the reproduction environment for any one paper. Read
`.llm/product_scope.md` for the normative product/publication boundary.

For a fresh-chat handoff, read `.llm/state.md` before using this map. That file records the current
implemented boundary and next increment; this file records where responsibilities live.

## Current state

Phases A through F3 are complete; Phase F4 is underway. The current public surface includes
`PiPLSRegression`,
`PiPLSPathCV`, `PiPLSComponentPath`, `PiPLSComponentResult`, `PiPLSDecomposition`,
`PiPLSValidationReport`, public selection metrics,
`StatisticalSupportWarning`, deterministic synthetic dataset generation, pure numerical
`pipls.inspection`, and optional Pi-PLS-specific and shared PLS-family figures under `pipls.plotting`.

The transparent real-dataset suite contains pulp, sugarcane, and tobacco. A licensing review of
the remaining companion-analysis candidates intentionally excluded Corn, the legacy Citrination
Steel table, SARCOS, and FRED-MD from this repository because the exact source materials do not
carry sufficiently clear redistribution rights. Public navigation now describes the installable
package, API, examples, datasets, validation, and releases.
The over-general synthetic manifest, universal result schema, and broad CI runner have been removed.
The benchmark layer implements all four focused synthetic questions as separate scripts. Pulp,
Sugarcane, and Tobacco are component-path examples rather than benchmark or test-suite executions.
`make examples` runs every numbered example explicitly, beginning with the literal-matrix
`01_minimal_fit_and_plot.py` quickstart and including the complete real-data analyses. Example 09
writes separate Pi-PLS and standard PLS (NIPALS) CSVs and derives the comparison PDFs from those
canonical tables. Pulp, Sugarcane, and Tobacco plot their Pi-PLS paths directly, keep OOF and
inspection results in memory, and write only final PDF figures. Pulp also exposes the conditional
predictor-rank profile at three components. Tobacco owns its full-SVD spectral figures, source-order
response pagination, and raw observation diagnostics directly.
No block-aware scaling API is designed or scheduled. Decision 0054 defines Python 3.10–3.14
support, guarded runtime dependency ranges, and a Python 3.10 minimum-dependency constraint
environment. Decision 0055 implements separate minimum, supported-Python, and latest-compatible CI
environments with resolved-version diagnostics. Decision 0056 adds clean installed wheel and
source-distribution validation. Decisions 0058--0061 make every public plotter a caller-embeddable
one-axis primitive, replace the former factor and prediction composites, and make the example layer
own every report figure, panel, legend, title, PDF operation, and close. Decision 0062
establishes the canonical Pulp workflow for the tutorial-first documentation phase. Decision 0063
generates its deterministic single-chart SVG assets and manifest. Decision 0064 makes the detailed
Pulp tutorial the primary pedagogical route through the served documentation, using checked source
snippets and one generated chart per explanation. Decision 0065 completes the transition by
separating tutorial, task-guide, scientific-reference, and generated-API ownership. Decision 0066
introduces the immutable concise component-path API. Decision 0067 makes Sugarcane direct, and
Decision 0068 makes the Pulp example and tutorial direct while removing the shared workflow wrapper.
Decision 0069 makes Tobacco direct and removes the final post-analysis table and OOF helpers. The
in-memory comparison rewrite is the current next increment.

## Implemented estimator and selection boundary

Decisions 0039 and 0040 are fully implemented. `PiPLSRegression` owns one explicit fixed rank
pair and no cross-validation or selection results. `PiPLSPathCV` owns the complete triangular-
selection lifecycle and defaults to the explicit complete-component sentinel `"all"`. Obsolete
private selection machinery and duplicate fitted aliases have been removed; Pi-PLS-specific output
is canonical in `decomposition_`. Supported pipelines infer their unique terminal Pi-PLS step and
carry their own output-container configuration through cloning and refit.

## Accepted analysis ownership

Decisions 0042 and 0045 define the staged fitted-model analysis surface. Pi-PLS-specific $P$, $D$,
and $Q$ inspection remains explicitly method-owned. Scores, loadings, coefficients, biplots,
observation diagnostics, and prediction diagnostics are shared PLS-family analyses with an
estimator-neutral API. Ordinary PLS remains the component-path and benchmark comparator. The numbered post-analysis examples apply shared tools only to the selected Pi-PLS model and fit no
final ordinary PLS model. Pulp, Sugarcane, and Tobacco own their OOF prediction, inspection,
figure composition, and final PDF output directly. Tobacco additionally owns source-order response
pagination and raw observation diagnostics.

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
- `src/pipls/plotting.py`: optional single-chart Matplotlib functions for explicit inspection
  results; callers own axes composition, legends, figure-level layout, and output.
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
- `examples/`: self-contained numbered user workflows with a literal-matrix quickstart first,
  followed by an explained synthetic train/test use case, one explicit comparison example, and
  complete Pi-PLS real-data analyses. Advanced splitters are documented rather than combined into a
  context-free numbered script.
  Underscore-prefixed `examples/_support/` contains only comparison-specific helpers rather than
  primary entry points. Example 09 owns the separate Pi-PLS and standard PLS (NIPALS) paths and
  comparison PDFs. Pulp, Sugarcane, and Tobacco own direct Pi-PLS paths, scikit-learn OOF
  predictions, immutable inspection results, and explicit final figures; Pulp additionally owns its
  conditional predictor-rank profile and tutorial snippets. Their pandas and Matplotlib
  requirements are grouped in the `examples` optional dependency extra.
- `benchmarks/`: four focused synthetic package-validation scripts, each with one minimal generated
  CSV. Real-data analyses are not duplicated here.
- `docs/`: the self-contained documentation source, including served user guides, implemented
  theory, generated-API source pages, release notes, local MathJax configuration, and maintainer
  records under `docs/decisions/`. Root `mkdocs.yml` defines the strict Material site and excludes
  the maintainer records from rendered pages and search.
- packaging and release configuration: installable distributions, compatibility policy, versioning,
  and release automation. `constraints/minimum.txt` records the maintainer-only lower-bound test
  environment; `.github/workflows/tests.yml` owns the three compatibility CI environments;
  `.github/workflows/build.yml` and `tools/check_distributions.py` own clean artifact-installation
  validation; `docs/compatibility.md` owns the public support statement.

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
- `docs/decisions/`: accepted maintainer history. These records remain versioned and distributed
  but are excluded from the served MkDocs site.

## Architectural invariants

- Runtime code does not import from `.llm`, tests, examples, docs, scripts, datasets, or benchmarks.
- Served Markdown under `docs/` is self-contained and does not link outside the documentation
  source tree. It defines Pi-PLS, component count, predictor rank, and the CV-MSE selection curve
  before specialized path or factorization terminology. `.llm` contracts and `docs/decisions/` may
  be more detailed but are never user prerequisites.
- The source distribution ships `mkdocs.yml`, the Makefile, documentation sources and assets, the
  minimum-dependency constraint file, and the helpers needed for clean strict documentation and
  installed-distribution checks. Generated `site/` output is excluded from Git, distributions, and
  snapshots.
- Runtime metadata and public documentation agree on Python 3.10–3.14 and the guarded NumPy,
  scikit-learn, and joblib ranges. CI separates the Python 3.10 minimum stack, normal resolution on
  every supported interpreter, and explicit latest-compatible upgrades on Python 3.14; every job
  prints the resolved interpreter and runtime dependency versions. The build workflow separately
  installs the wheel and source distribution into clean environments and verifies public runtime
  behavior outside the checkout.
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
make dist-check
```

Use `make examples` whenever numbered examples or their generated application artifacts change.
Use `make build` for a quick artifact build, and use `make dist-check` whenever packaging,
dependencies, public modules, or included data files change.
Record each validation target as passed, failed, or not run.
