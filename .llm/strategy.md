# Pi-PLS package development strategy

## Status and purpose

This is the operational development strategy for the long-lived `pipls` software repository. It
records the accepted architecture as small, testable increments that can be maintained during
LLM-assisted development. Current decision records and normative `.llm` contracts replace private
or historical planning materials as the maintained source of project intent.

The repository-product boundary is normative in `.llm/product_scope.md`: `pipls` owns package
functionality, user documentation, examples, datasets, lightweight validation benchmarks, tests,
packaging, and releases. Paper-specific reproduction belongs in downstream repositories that pin
tagged `pipls` versions.

## Ownership

- The project owner decides scientific aims, product scope, publication boundaries, public API
  commitments, and unresolved methodological choices.
- The LLM maintainer owns the consistency of this operational strategy with the repository. A
  patch that changes phases, architecture, public behavior, mathematical contracts, or workflow
  must update this file when the strategy is affected.
- Git history, tests, and implemented behavior remain authoritative evidence of what exists.
  This strategy describes intended work; it must not claim an increment is complete before its
  acceptance tests pass.
- The LLM maintainer may clarify, split, reorder, or mark work complete, but must not silently
  change a scientific or public-interface decision fixed by the project owner.

## Development principles

1. Work in small increments that can be tested and committed independently.
2. Port trusted behavior before adding convenience features.
3. Keep the fixed-parameter numerical core separate from estimator and model-selection layers.
4. Route public rank and path searches through shared private candidate-evaluation and
   rank-search engines.
5. Add focused tests with every behavioral change.
6. Compare subspaces, regression maps, and predictions rather than raw singular-vector signs.
7. Fit every learned preprocessing operation inside the corresponding training fold.
8. Keep paper-specific orchestration outside the package repository; downstream reproduction
   repositories pin released package versions.
9. Keep real-data input transparent: examples read `X` and `Y` explicitly and do not depend on
   a public registry, generic loader, or required metadata sidecar.
10. Update `.llm/state.md`, the relevant `.llm` contracts, and user-facing documentation when a
   phase, roadmap, architecture, or public contract changes.
11. Return one root-relative Git patch per increment with an explicit validation report.
12. Use direct `git apply`, `git add`, and `git commit` commands; do not maintain wrapper scripts
    for patch application or committing.
13. Do not combine algorithm porting, API expansion, dataset migration, benchmark freezing, and
    repository-product cleanup in one patch unless the dependency cannot be separated.
14. Keep repository tests durable: verify behavior and file structure, not current roadmap prose
    or documentary metadata values.
15. Preserve the implemented estimator-internal, fold-local centering/scaling contract. Do not
    design or anticipate future block-aware standardization APIs until the project owner starts a
    dedicated design phase.

## Fixed architectural decisions

- Runtime package layout: `src/pipls/`.
- Public estimator style: scikit-learn compatible.
- Public names: `n_components`, `predictor_rank`, and
  `samples_per_predictor_rank`; no public aliases `h`, `r_pi`, or `c`.
- Standard response subspace for the current Pi-PLS construction: leading right singular
  vectors of `Z.T @ Y`.
- Predictor-rank modes target four semantics: integer, `"max"`, `"optimal"`, and `"auto"`.
- `"optimal"` means exhaustive CV evaluation of every admissible predictor rank.
- `"auto"` means deterministic adaptive coarse-to-fine search and is explicitly approximate.
- General search support uses the total number of observations supplied to `fit()`, while centered
  training-fold sizes remain hard feasibility caps. The ordinary public defaults are
  `samples_per_predictor_rank=5` and `cv=5`.
- All CV-based selection uses fold-local response-standardized MSE.
- Search approximation and linear-algebra approximation are separate policies. Randomized SVD
  must be introduced through an explicit solver contract and diagnostics, not hidden inside the
  meaning of `predictor_rank`.
- `PiPLSRegression` always centers `X` and `Y`; with `scale=True` it also divides both blocks by
  safe training-sample standard deviations, while `scale=False` retains centering.
- Every candidate fit learns those statistics from its own training fold, and the selected model
  refits them on the complete training set supplied to `fit()`.
- Custom learned preprocessing is searched around the complete supported estimator boundary.
- `PiPLSPathCV` supports a direct `PiPLSRegression` or a `Pipeline` whose final step is
  `PiPLSRegression`; arbitrary nested meta-estimators are not implied.
- Publication-specific rank rules and reporting conventions are not package defaults; they
  belong in downstream reproduction repositories.
- D2 metadata support is deliberately limited to `groups` for splitters. Weighted fitting,
  `sample_weight`, and general-purpose metadata routing are out of scope.
- Real-data users supply `X` and `Y` directly. No public registry, generic loader, required
  metadata sidecar, or hidden example I/O utility is part of the accepted architecture.
- `.llm/` is tracked repository infrastructure and is excluded from the installable package.
- Patch application and commits use ordinary Git commands rather than project wrapper scripts.
- Snapshots contain repository-root contents without an enclosing project directory.
- Patches are unified Git patches relative to repository root.
- Living `.llm` documents and dataset metadata are reviewed artifacts, not duplicated as fixed
  phrase or field-value assertions in the test suite.
- `pipls` is a long-lived software-product repository rather than a manuscript-reproduction
  repository.
- Future block-aware standardization remains a valid product direction, but only its API, block
  semantics, naming, scheduling, and implementation are deferred. Current estimator-internal
  centering/scaling is implemented and normative.

## Increment sequence

### Phase A: repository and specification foundation

Acceptance conditions:

- repository skeleton, packaging metadata, Makefile, and CI exist;
- `.llm` contracts, snapshot creation, optional patch export, direct-Git workflow, and
  navigation are tested;
- snapshot members are rooted at repository root;
- no provisional estimator exists.

Current status: **complete**. The repository foundation, contracts, snapshot layout, direct-Git
workflow, and workflow tests are established.

### Phase B1: fixed-parameter private core

Implement only the theory-faithful fixed-$(h,r_\pi)$ numerical kernel in `src/pipls/_core.py`.

Required outputs:

- explicit input validation and admissibility checks;
- a small internal result structure with documented shapes;
- deterministic tests of dimensions, orthogonality, nonnegative ordered dilation values,
  prediction factorization, rank-deficient inputs, and $p \gg n$ inputs;
- frozen comparisons with the trusted reference implementation;
- updates to `.llm/mathematics.md` and `.llm/numerical_contracts.md` where necessary.

Do not add the public estimator, automatic rank selection, datasets, or plotting in this
increment.

Current status: **complete**. The fixed-parameter private core and its reference, invariant, and
boundary tests are committed.

### Phase B2: fixed-rank public estimator

Add `PiPLSRegression` for an explicit integer `predictor_rank` only.

Required outputs:

- scikit-learn constructor, validation, fitted-state behavior, `fit`, `predict`, `transform`, and
  scalar `score`;
- documented centering and scaling semantics;
- coefficient orientation and fitted attributes consistent with `.llm/public_api.md`;
- estimator-contract and pipeline smoke tests.

Current status: **complete**. The public explicit-integer estimator, preprocessing semantics,
package export, pipeline smoke tests, and strict typing boundary are committed.

### Phase C1: rank-bound helper and `predictor_rank="max"`

Add one tested fold-safe rank-bound helper and the rule-fixed estimator mode. Do not add internal
cross-validation in the same increment.

Current status: **complete**. The shared ceiling-based bound and the non-CV `"max"` estimator mode
are committed and tested.

### Phase C2a: private CV-selection primitives

Freeze the reusable split, candidate-grid, fold-local response-scale, response-standardized loss,
and deterministic tie-breaking contracts before changing public estimator behavior.

Current status: **complete**. The reusable split, candidate-grid, fold-local response-scale,
response-standardized loss, and deterministic tie-breaking primitives are committed and tested.

### Phase C2b: `predictor_rank="auto"`

Integrate the Phase C2a primitives into `PiPLSRegression` for one fixed `n_components`, including
fold-local preprocessing, diagnostics, standard scorer orientation, optional parallel candidate
evaluation, and full-data refitting.

Current status: **complete**. Automatic mode is the estimator default, uses the smallest
materialized training fold for its candidate bound, stores split and mean diagnostics, and refits
the selected fixed-rank model on all supplied data.

### Phase C2c: split exhaustive and adaptive rank-search semantics

Replace the provisional exhaustive meaning of `predictor_rank="auto"` with the accepted public
search-policy split:

- `predictor_rank="optimal"`: exhaustive evaluation of every admissible rank;
- `predictor_rank="auto"`: deterministic adaptive logarithmic coarse-to-fine evaluation with a
  final exhaustive search over a small integer interval;
- integer and `"max"`: unchanged.

The implementation must cache candidate results, reuse one materialized split set, preserve
fold-local preprocessing, expose evaluated-rank diagnostics, and document that `"auto"` is not
guaranteed to recover the exhaustive optimum on an arbitrary non-unimodal CV curve. Because the
package is pre-alpha, the old exhaustive `"auto"` behavior is renamed without a compatibility
alias.

Current status: **complete**. Exhaustive search is exposed as `"optimal"`; adaptive `"auto"`
uses deterministic logarithmic batches, cached evaluations, neighbor-bracket refinement, a final
exhaustive interval of at most 10 ranks, and reconstructable diagnostics.

### Phase C2d: scalable linear-algebra policy

Add an explicit `svd_solver` policy, initially supporting `"full"`, `"randomized"`, and `"auto"`,
with a deterministic `random_state` contract and fitted solver diagnostics. Keep this independent
from predictor-rank search semantics so users can distinguish exhaustive versus adaptive search
from exact versus approximate SVD.

Current status: **complete**. The estimator now exposes `svd_solver={"full", "randomized",
"auto"}` and `random_state`; automatic selection follows the conservative matrix-size and
retained-rank rule, only the predictor SVD may be randomized, and fitted plus CV-fold solver
diagnostics are available.

### Phase C2e: public parameter-validation hardening

Audit exposed constructor controls at the estimator boundary. Reject invalid and degenerate
integer-like values consistently, preserve deliberate NumPy-scalar support, define the public
random-seed range, warn when the rule parameter supplies fewer than five training samples per
retained predictor-rank direction, and add exhaustive API-level tests.

Current status: **complete**. Constructor validation now runs before dependency calls; integer,
boolean, CV, job-count, seed, solver, scorer, and extreme rank-bound cases have API-level tests;
`StatisticalSupportWarning` is public; and extremely small positive rule parameters saturate
safely without overflow.

### Phase D1: complete path analysis

Add `PiPLSPathCV` with the admissible triangular grid, shared materialized splits, standard
scikit-learn scorer orientation, and complete-pipeline cloning. Its search-policy interface must
mirror the established `"optimal"` and `"auto"` distinction rather than inventing a second set of
meanings.

Current status: **complete**. `PiPLSPathCV` is public, evaluates exhaustive or adaptive
triangular paths, clones complete pipelines inside every fold, infers or validates the nested
Pi-PLS parameter prefix, exposes standard and Pi-PLS-specific diagnostics, and refits the globally
selected complete estimator.

### Phase D1a: shared private selection engine

Extract fold-level candidate evaluation and one-dimensional rank-search orchestration so
`PiPLSRegression` and `PiPLSPathCV` share the same private machinery without making either
public class wrap the other. Add equivalence tests showing that one fixed path row matches the
corresponding estimator rank search.

Current status: **historically complete and partially superseded by Decision 0039**. Candidate
cloning, fold-local fitting, scoring, standardized loss, caching, and optional solver diagnostics
remain centralized in `_cv_engine.py`, while exhaustive and adaptive rank refinement remain in
`model_selection.py`. `PiPLSRegression` no longer consumes this machinery; it is now path-owned and
scheduled for cleanup in Phase F1 patch 4.

### Phase D1b: scikit-learn and PLS-style API alignment

Align both public classes with estimator-aware validation, feature-name and output-container
contracts, PLS-style score/weight/loading attributes, standard search-result names, and a public
structured decomposition result. Preserve pandas and other indexable containers through complete-
pipeline path folds and expose the selected nested estimator without flattening coefficients.

Current status: **complete**. Both public classes pass all applicable common estimator checks;
only the tuple-valued cross-decomposition transformer checks are declared as expected failures,
matching the special behavior of PLS estimators. `PiPLSDecomposition`, `best_pipls_`, standard CV
result aliases, fit-time copy semantics, feature names, pandas output, and container-preserving
path folds are implemented and tested.

### Phase D1c: final scikit-learn cleanup boundary

Complete the remaining pre-validation compatibility work: inverse reconstruction, canonical
read-only decomposition arrays, standard CV sentinels and timings, conditional path delegation,
an explicit supported estimator boundary, and minimum-version CI.

Current status: **complete**. `PiPLSRegression.inverse_transform`, canonical decomposition aliases,
`cv=None`, `scoring=None`, `scorer_`, timing diagnostics, conditional delegated methods, direct or
terminal-pipeline path support, and minimum-version scikit-learn CI are implemented and tested.

### Phase D2: LOO and advanced split protocols

Add ordered out-of-fold predictions, the documented LOO protocol, grouped and temporal examples,
and explicit selection-conditioned reporting. Use ordinary scikit-learn splitters and preserve a
narrow metadata contract: explicit `groups` only, with no weighted fitting or general sample
metadata routing.

Current status: **complete**. Both public interfaces support explicit group metadata, ordinary
repeated, predefined, temporal, and LOO splitters, optional row-ordered OOF predictions with
repeat counts and partial coverage, singleton-safe scoring rules, pooled OOF R2 diagnostics, and
immutable selection-conditioned validation reports. Weighted fitting is not supported.

### Phase E1: dataset schema and deterministic synthetic generator

Introduce the dataset boundary before any real-data migration.

Acceptance conditions:

- one validated dataset container for arrays, names, sample identifiers, and metadata;
- explicit shape, dtype, finite-value, naming, and provenance rules;
- a side-effect-free seeded synthetic generator;
- configurable shared, predictor-specific, and response-specific latent structure;
- configurable ranks, strengths, scales, distributions, and noise;
- deterministic train/test generation without preprocessing leakage;
- focused invalid-input, invariant, and reproducibility tests;
- no real dataset migration or network access in this increment.

Current status: **complete**. `PiPLSDataset` validates and freezes arrays, names, sample IDs, provenance, metadata, and optional truth. `make_pipls_regression` and `make_pipls_train_test` provide local seeded generation with configurable latent roles, strengths, distributions, scales, and noise. Unit, API, invariant, and reproducibility tests cover the boundary. No real dataset or network access was added.

### Phase E2: transparent real-data input contract

Fix the programming-user and example contract before migrating real datasets.

Acceptance conditions:

- plain arrays or data frames supplied as `X` and `Y` remain the primary real-data interface;
- `PiPLSDataset` is optional and no metadata sidecar is required for fitting;
- no public registry or generic real-data loader is planned;
- examples show ordinary reading, alignment, column selection, and matrix construction directly
  rather than hiding them behind utilities;
- repository-specific provenance and preparation information may be tracked for reproducibility
  without becoming runtime requirements for external users;
- the revised contract is navigable from a fresh snapshot.

Current status: **complete**. `.llm/data_io.md` defines the transparent input and example policy,
and the roadmap now proceeds directly to one explicit real-dataset integration per patch.

### Phase E3: real dataset integrations

Migrate and validate one dataset per coherent increment. Every committed dataset uses
comma-delimited `X.csv`, comma-delimited `Y.csv`, and a consistent documentary `metadata.yaml` as
defined in `.llm/dataset_layout.md`. Record only public or included sources, citation, license,
redistribution decision, preparation choices, shapes, columns, row ordering, missing-value policy,
and integrity hashes. Make the analysis example read `X` and `Y` directly with ordinary NumPy or
pandas code. Do not introduce a generic registry or loader, private-source references, or
preparation-only scripts. Defer Corn reconstruction until its preprocessing choices are explicitly
fixed; Corn will expose its public raw-data reading and analysis-relevant preprocessing.

Current status: **complete for the current reference suite**. Pulp provides a compact named
multivariate process dataset; sugarcane adds a 1,721-column regular wavelength grid; and tobacco
adds 347 samples, 1,557 raw FT-NIR predictors, and 13 responses. Every integration uses public
provenance, direct `X.csv`/`Y.csv` reading, documentary metadata, and no runtime loader or hidden
preparation utility. Decision 0033 removed the former Linnerud integration because it no longer
served a useful package-level example or validation role.

### Product transition P1: repository cleanup

Remove the transitional `paper/` and `scripts/reproduce_paper/` placeholders and rewrite public
repository navigation around the installable software product. Preserve historical scientific
context where it explains accepted behavior, but remove future promises that manuscript figures,
complete comparison grids, or paper orchestration will be implemented inside `pipls`.

Acceptance conditions:

- no runtime or public API behavior changes;
- paper-oriented placeholders are removed;
- the public README and documentation navigation describe package installation, use, examples,
  datasets, validation, and releases;
- publication-specific reproduction is described only as downstream work that pins a tagged
  `pipls` version;
- no synthetic benchmark result or new block-aware scaling API is introduced in the same patch.

Current status: **complete**. The paper-reproduction placeholders have been removed, and public
navigation now describes the installable package, user documentation, examples, datasets,
validation, and release responsibilities.

### Phase E4a: focused synthetic benchmark design

Define package benchmarks as separate user-facing questions rather than a universal experiment
framework.

Acceptance conditions:

- every benchmark states one question, controlled setup, method role, minimal metrics, dedicated CSV
  output, and interpretation boundary;
- fixed-structure recovery, rank selection, predictor-nuisance comparison with ordinary PLS, and
  solver consistency remain separate benchmarks;
- result columns are included only when they answer the benchmark question;
- software versions, execution controls, timing, and memory are omitted unless the benchmark is
  explicitly about compatibility or resources;
- ordinary PLS is the sole planned external comparator;
- OLS, CCA, publication grids, and figure generation remain outside the repository;
- no universal manifest, universal schema, or broad runner is introduced.

Current status: **complete**. Decision 0030 removes the earlier universal manifest/schema/runner
architecture and establishes one question, one script, and one minimal CSV output per benchmark.

### Phase E4b: focused benchmark implementation

Implement the accepted benchmarks one at a time so each patch remains scientifically and
operationally reviewable.

Current status: **complete**. Fixed-structure recovery, adaptive rank selection, paired
predictor-nuisance comparison with ordinary `PLSRegression`, and full-versus-randomized solver
consistency are implemented as independent scripts with minimal question-specific CSV outputs.

### Phase E4c: representative real-data examples — complete

Pulp, Sugarcane, and Tobacco provide transparent component-path analyses with direct `X.csv` and
`Y.csv` reading. Each example writes canonical Pi-PLS and standard PLS CSVs, derives a comparison
PDF from those tables, and performs a separate fixed Pi-PLS fit after an explicit component choice.

The initial real-data smoke benchmark scripts and tests were removed after review because they
repeated the same Pi-PLS paths, while dataset-specific tests executed the complete examples again.
Default package validation now protects the reusable machinery on small data and the durable
repository/file contracts. `make examples` runs every numbered example explicitly when application
artifacts or end-to-end example validation are needed.

### Phase F1: fixed-estimator and path-search correction

Decision 0039 establishes a staged API correction before release hardening. The target boundary is:

- `PiPLSRegression` fits one explicit fixed pair and owns no cross-validation or search results;
- `PiPLSPathCV` owns the bounded triangular scan and conditional predictor-rank selection;
- the path ceiling uses `samples_per_predictor_rank=5`;
- direct fixed fits warn when $n/r_\pi<4$;
- expected support warnings are suppressed only inside path-controlled feature probes, candidate
  fits, optional OOF fits, and the selected full-data refit;
- examples continue to recommend `PiPLSPathCV`, not a hand-built `GridSearchCV` surface.

Implementation order is fixed-model estimator, sole path selection, dead-code consolidation,
example and guide alignment, then final API/minimality audit.

Current status: **patches 2 and 3 complete; obsolete private selection cleanup next**.

### Product documentation and release hardening

Build a user-oriented documentation surface, API reference, compatibility policy, clean-install and
build checks, licensing audit, release notes, and versioned releases. These are continuing software
product responsibilities rather than the final steps of one publication.

Current status: **planned and ongoing**.

### Current standardization and deferred block-aware variants

Estimator-internal standardization is implemented now and is part of the model-fitting contract.
Each `PiPLSRegression` fit centers `X` and `Y`; `scale=True` additionally uses safe sample standard
deviations estimated from that fit's training observations. Cross-validation clones and fits the
complete estimator inside each training fold, and the selected model refits on all supplied
training data. This behavior mirrors the leakage-safe role of standardization in
`PLSRegression`.

Future block-aware standardization remains a valid product direction expected months from now.
Only that extension is deferred. This strategy deliberately defines neither whether it is
estimator-owned or represented in a supported model pipeline, nor any class names, constructor
parameters, block semantics, schedule, or implementation sequence. Any future variant must learn
its scaling statistics inside each training-fold fit and the final full-training refit; it must
never be prefit globally before cross-validation.

Current status: **current estimator standardization complete; block-aware API design deferred**.

## Current next increment

Implement patch 4 of Phase F1: remove obsolete private selection machinery and duplicated result
structures left by the former embedded `PiPLSRegression` search. Preserve the public behavior of
the fixed estimator and path meta-estimator.

Resume documentation and release hardening only after the full Phase F1 sequence and final audit.

## Maintenance protocol

For every patch, the LLM maintainer should:

1. read `.llm/README.md`, this file, and the relevant contracts;
2. identify the current phase and avoid work assigned to later phases;
3. state which acceptance conditions the patch addresses;
4. update `.llm/state.md`, `Current status`, and `Current next increment` when phase state changes;
5. update fixed decisions only after an explicit owner decision;
6. run and report each applicable Makefile validation target;
7. review tests against `.llm/testing.md` and remove accidental coupling to living prose or
   documentary field values;
8. return a root-relative patch and provide the exact direct Git commands for checking, applying,
   inspecting, staging, and committing it.
