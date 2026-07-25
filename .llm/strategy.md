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
10. Keep numbered examples self-contained and user-oriented: each demonstrates a recognizable use
    case, comparison, or benchmark, explains its data, labels its output, and assumes no publication
    context.
11. Update `.llm/state.md`, the relevant `.llm` contracts, and user-facing documentation when a
    phase, roadmap, architecture, or public contract changes.
12. Return one root-relative Git patch per increment with an explicit validation report.
13. Use direct `git apply`, `git add`, and `git commit` commands; do not maintain wrapper scripts
    for patch application or committing.
14. Do not combine algorithm porting, API expansion, dataset migration, benchmark freezing, and
    repository-product cleanup in one patch unless the dependency cannot be separated.
15. Keep repository tests durable: verify behavior and file structure, not current roadmap prose
    or documentary metadata values.
16. Preserve the implemented estimator-internal, fold-local centering/scaling contract. Do not
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

Current status: **complete**. The estimator exposes `svd_solver={"full", "randomized", "auto"}`
and conventional integer, NumPy `RandomState`, or `None` random-state forms; automatic selection
follows the conservative matrix-size and retained-rank rule, only the predictor SVD may be
randomized, and resolved solver diagnostics are available through `decomposition_`.

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

Current status: **historically complete and superseded by Decision 0039**. Candidate cloning,
fold-local fitting, scoring, standardized loss, and caching remain path-owned. Unused solver
diagnostics, duplicate candidate metadata, and former estimator-search helpers have been removed.
Exhaustive and adaptive rank refinement remain in `model_selection.py`.

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

Current status: **complete**. `PiPLSRegression.inverse_transform`, canonical read-only
`decomposition_`, `cv=None`, `scoring=None`, `scorer_`, timing diagnostics, conditional delegated
methods, direct or terminal-pipeline path support, and minimum-version scikit-learn CI are
implemented and tested. Decision 0040 later removes duplicate decomposition aliases and makes
refit-dependent path method availability explicit.

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
preparation-only scripts. Before implementation, verify that the exact source material has an
explicit source-level license or permission granting redistribution and adaptation for general
repository use; public download access alone is insufficient.

Current status: **complete for the current reference suite and licensing audit**. Pulp provides a
compact named multivariate process dataset; sugarcane adds a 1,721-column regular wavelength grid;
and tobacco adds 347 samples, 1,557 raw FT-NIR predictors, and 13 responses. Every integration uses
public provenance, direct `X.csv`/`Y.csv` reading, documentary metadata, and no runtime loader or
hidden preparation utility. Decision 0033 removed the former Linnerud integration because it no
longer served a useful package-level example or validation role. Decision 0041 intentionally
excludes Corn, the legacy Citrination Steel table, SARCOS, and FRED-MD because the exact source
materials do not provide sufficiently clear redistribution rights. No legacy dataset remains
pending.

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
`Y.csv` reading. Example 04 owns the Pi-PLS and standard PLS comparison figures; examples 05–06
perform separate Pi-PLS-only fixed fits after explicit component choices, while example 07 uses the
stored one-standard-error recommendation for its final fixed fit. Phase F4 subsequently makes
all four real-data workflows direct in-memory analyses.

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
- direct fixed fits warn when $n/r_\pi<3$;
- expected support warnings are suppressed only inside path-controlled feature probes, candidate
  fits, optional OOF fits, and the selected full-data refit;
- examples continue to recommend `PiPLSPathCV`, not a hand-built `GridSearchCV` surface.

Implementation order is fixed-model estimator, sole path selection, dead-code consolidation,
example and guide alignment, then final API/minimality audit.

Current status: **complete**. The final audit removed the redundant path parameter-prefix control,
confirmed fixed-pair `GridSearchCV` interoperability, and was followed by Decision 0040's final
public-surface polish: `n_components_values="all"`, conventional random-state forms, a reusable
public scorer callable, refit-dependent method availability, and canonical decomposition-only
Pi-PLS diagnostics. Decision 0102 later changes only the constructor presentation and defaults:
selection-only `refit=False` and a stable package scorer name resolving to that callable.

### Phase F2: model inspection and post-analysis

Decision 0042 separates component-path selection diagnostics, fitted-model interpretation, and
prediction diagnostics. Implement a reusable but restrained analysis surface without changing the
Pi-PLS numerical core, fixed estimator, or path-selection engine.

Acceptance conditions:

- pure numerical computations and immutable results live in `pipls.inspection` without pandas or
  Matplotlib;
- rendering remains outside the runtime package; examples use ordinary Matplotlib from immutable
  inspection results and perform no analytical file round trips;
- Pi-PLS decomposition views display $P$, $D$, and $QD$ while preserving $PDQ^\mathsf{T}$ under
  deterministic display-only sign canonicalization;
- prediction diagnostics accept explicit predictions, use residuals $e=y-\hat y$, standardize
  from supplied observed responses with `ddof=1`, and record prediction provenance;
- established PLS coverage begins with scores, X/Y loadings, coefficients, one low-dimensional
  biplot, and raw observation diagnostics;
- real-data examples own OOF prediction, physical-axis semantics, pagination, and report
  composition; the F2 table/CSV implementation is subject to the later F4 in-memory simplification;
- complete real-data post-analysis remains under `make examples`, while `make check` uses small
  synthetic contracts;
- VIP, automatic variable selection, confidence ellipses, uncertainty intervals, permutation
  tests, theoretical outlier limits, and contribution plots remain deferred.

Decision 0045 corrected the model-ownership boundary. The estimator-neutral shared inspection API,
Pi-PLS-only numbered-example post-analysis, quantity-based artifacts, and static boundary
enforcement are complete. Decisions 0079--0082 later remove the plotting API in favor of direct
Matplotlib rendering from immutable results.
Decision 0046 then reduced the numbered scripts to their scientific stages, moved repository-layout validation out of the scripts, and made the required result directories tracked repository structure.

Current status: **complete**.

### Phase F3: atomic plotting composition

Decision 0058 refines the pre-release plotting surface around ordinary Matplotlib composition.
Every public single-chart plotter draws on one `Axes`, accepts a caller-supplied axis, and returns
`(figure, axis)`. Plotters may provide semantic labels and titles, but callers own legends, subplot
geometry, figure-level layout, saving, display, and closing. The package must not preserve
one-entry axis dictionaries or package-owned multi-panel figures in the first release.

Implementation order:

1. add the common axis-resolution contract to the existing atomic PLS-family plotters;
2. split the Pi-PLS factor composite into separate $P$, $D$, $Q$, and $QD$ functions;
3. split the prediction-diagnostic composite into separate observed/predicted, residual, and RMSE
   functions;
4. migrate numbered examples and documentation so all tiled layouts are caller-owned and generated
   PDFs preserve their intended scientific groupings.

Current status: **complete**. Decisions 0058--0061 establish the one-axis plotting contract, split
the Pi-PLS factor and prediction-diagnostic composites, and make the example layer create every
report figure and axis. The real-data reports use caller-owned factor, prediction, and shared
latent-model panels; coefficient curves remain full-width pages.

### Phase F4: pre-release result and example simplification

Reduce the remaining application-layer indirection before release. Public numerical results should
be directly accessible as arrays or immutable objects, and numbered examples should operate on
those results in memory rather than using generated CSV files as analytical or plotting
intermediates. Committed dataset tables and final focused benchmark CSV products remain outside
this restriction.

Implementation order:

1. immutable component-path API;
2. direct Sugarcane reference workflow;
3. direct Pulp example and tutorial;
4. direct Tobacco workflow and removal of post-analysis table machinery;
5. in-memory Pi-PLS/PLS comparison;
6. package-wide cleanup, documentation migration, and structural enforcement.

Current status: **complete**. Decision 0066 adds `PiPLSComponentPath`,
`PiPLSComponentResult`, and `PiPLSPathCV.component_path_`. Decisions 0067–0069 make Sugarcane,
Pulp, and Tobacco direct and remove their analytical CSV/report machinery. Decision 0070 makes the
Pi-PLS/PLS comparison direct, replaces its DataFrame result with immutable arrays, and removes the
CSV plotting helper. Decision 0071 removes duplicate matrix-path aliases, makes `cv_results_` the
sole detailed candidate surface, and enforces the direct in-memory example policy structurally.
Decision 0072 adds a derived immutable one-component predictor-rank profile so ordinary inspection
does not require manual `cv_results_` masking while preserving that dictionary as the source of
truth. Decision 0106 adds a derived fold-based CV-MSE standard error to the concise path and profile
records without changing their stored arrays, selection semantics, or pickle payloads. Decision
0107 adds exact stored-value minimum-CV-MSE and one-standard-error recommendation methods to
`PiPLSComponentPath`; they return immutable scalar rows and do not fit, refit, or mutate search
state. Decision 0108 makes example 07 the single maintained application of the one-standard-error
method.

### Product documentation and release hardening

Build a user-oriented documentation surface, API reference, compatibility policy, clean-install and
build checks, licensing audit, release notes, and versioned releases.

Decision 0049 establishes `docs/` as the self-contained public source, adds public example and
decision navigation, makes the theory guide independent of `.llm`, and removes unused alternatives
from current documentation.

Current status: **documentation Patches D1--D4 complete**. Decisions 0049--0065 establish the
self-contained documentation, generated
reference, compatibility validation, atomic plotting, and canonical Pulp workflow. Decision 0074
removes redundant task guides. Decision 0075 adds the short synthetic selection-and-prediction
tutorial, Decision 0076 repositions the Pulp tutorial around real-data selection qualification,
selection-conditioned OOF analysis, and representative plots, and Decision 0077 reduces the
README while moving maintenance ownership to `CONTRIBUTING.md` and project-validation navigation.
Phase F4 has completed the owner-approved pre-release result and example simplification.
Decision 0078 completes the reference cleanup with a result-object map, troubleshooting,
link-and-anchor validation, and documentation tests that avoid freezing living prose. Decision
0084 completes the focused Pulp prediction section by displaying the residual-versus-predicted
figure represented by the middle axis of the maintained three-panel snippet.

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

## Documentation foundation status

Decision 0050 completes the MkDocs foundation: the dedicated `docs` extra installs MkDocs and
Material; `make docs` performs a strict build; PyMdown Extensions and MathJax render the existing
notation; navigation follows the user journey; and generated `site/` output is ignored and cleaned.
Decision 0051 adds mkdocstrings and generated pages for every supported top-level object, with
source docstrings covering signatures, parameters, fitted attributes, shapes, and conditional
refit and OOF behavior. Decision 0052 originally completed the generated reference for inspection, plotting, datasets, and
metrics. Decision 0082 removes the plotting module and page; exact submodule coverage now applies to
inspection, datasets, and metrics. Decision 0053
validates the strict site in CI from both the checkout and a clean installation of the unpacked
source distribution.

Current status: **complete**. Decisions 0049--0053 define the self-contained source, strict site
build, generated public reference, distributed documentation inputs, and CI validation. Decision
0057 excludes maintainer decision records from the served site and requires concise, theory-linked
onboarding before path and factorization terminology. Decision 0090 consolidates the programming
reference into eight navigation entries while preserving generated coverage, stable interpretation
anchors, and all advanced selection and validation contracts.

## Compatibility policy status

Decision 0054 defines Python 3.10–3.14 as the supported interpreter range, retains an open-ended
`requires-python = ">=3.10"`, guards the next NumPy, scikit-learn, and joblib major versions, and
records one reproducible minimum stack in `constraints/minimum.txt`. Decision 0055 separates the
minimum, supported-Python, and latest-compatible CI responsibilities and requires resolved-version
diagnostics in every job. Decision 0056 builds the wheel and source distribution once, installs
each artifact into a separate clean environment outside the checkout, and runs one shared public
runtime smoke test with metadata and import-origin checks.

Current status: **compatibility policy, CI matrix, and installed-distribution validation complete**.
Decision 0073 additionally hardens the public estimator boundary with transactional fits,
range-safe preprocessing fallbacks, safe `copy=False` handling, and finite-result enforcement
without expanding the private numerical core.

### Pre-release audit hardening

An owner-directed package audit identified six focused corrections to complete before a separate
human review. They must be implemented and snapshotted one at a time:

1. clean committed-tree snapshot integrity;
2. fold-level numerical-rank feasibility in path selection;
3. consistent tolerant score ranking;
4. core public-result invariants;
5. inspection-result invariants and numerical safety;
6. dataset metadata closure.

Decision 0091 completes the first increment by replacing worktree copying with a clean `HEAD`
archive and tests for modified, staged, nonignored untracked, and ignored files. Decision 0092
completes the second increment by capping the path with the minimum rank verified after fold-local
preprocessing. The third increment uses one reference-anchored tolerant comparison for selection
and score ranking, so rank groups cannot expand through adjacent near-ties. Decision 0093 completes
the fourth increment by applying one direct-construction, defensive-copy, scalar, OOF-coverage, and
pickle-validation policy to the core estimator and path result records. Decision 0094 completes the
fifth increment by applying the same boundary to inspection records and using range-safe
calculations that reject unrepresentable derived values rather than returning nonfinite arrays.
The sixth increment closes the dataset boundary by rejecting object-dtype metadata arrays while
retaining copied read-only non-object arrays and recursively frozen heterogeneous mappings and
sequences. Release preparation and publication are explicitly outside this series and remain
deferred until the owner completes a human audit.

Current status: **all six pre-release hardening increments complete**.

### Owner-authorized human-audit continuation

The owner identified three additional focused increments during the human audit:

1. require the fixed `PiPLSRegression` rank pair explicitly and explain replaceable path-template
   seed values;
2. align notation and path-ceiling documentation and improve inspection navigation;
3. add one focused small-sample synthetic leave-one-out example.

Decision 0095 completes the first increment. Decision 0096 completes the second by separating
mathematical $Y$ from scikit-learn `y`, defining both resolved path ceilings before policy details,
using response-neutral residual labels, and grouping inspection concepts with the generated API.
Decision 0097 completes the third with one focused small-calibration leave-one-out workflow,
ordered OOF reporting, and an explicit pooled-versus-foldwise $R^2$ distinction. Release
preparation and publication remain outside this continuation.

Current status: **all three human-audit continuation patches complete**.

### Owner-authorized maintainer command orientation

The owner identified the flat Make target list as a contributor-entry problem. Decision 0098
retains all target names and recipes, identifies `make install` and `make check` as the primary
route, groups the remaining commands by task, and mirrors that hierarchy in `CONTRIBUTING.md`.

Current status: **maintainer command orientation complete**.

### Owner-authorized new-user onboarding series

A whole-package audit from a new user's perspective identified six focused increments:

1. remove committed example outputs and make source-distribution examples runnable;
2. establish the rendered documentation site as the primary user entry;
3. make numbered examples continuous and improve the first example's terminal output;
4. align path-search defaults and scorer representation with the documented workflow;
5. simplify optional dependencies and separate user installation from contributor setup;
6. refine the landing page and tutorial openings after the preceding contracts settle.

Decision 0099 completes the first increment by committing and snapshotting only example-result
placeholders, distributing those placeholders, and running example 01 from an extracted source
distribution. Decision 0100 completes the second increment with strict documentation validation on
pushes and pull requests and repository-derived GitHub Pages deployment from `master`. Decision
0101 completes the third increment by renumbering all maintained examples continuously from 01 to
07, migrating every active consumer, and labeling example 01 terminal output. Decision 0102
completes the fourth increment by making path evaluation selection-only by default and replacing the
function-object scoring default with a stable package name that resolves to the same public callable.
Decision 0103 completes the fifth increment by limiting optional dependencies to maintained
workflows, removing unused packages, and separating noneditable user installation from editable
contributor setup. Decision 0104 completes the sixth by adding restrained application-oriented
motivation and moving tutorial source and renderer provenance to terminal reproduction sections.
Package release preparation and Python-package publication remain outside this series.

Current status: **all six new-user onboarding patches complete**.

### Documentation and implementation alignment

Decision 0105 completes a cross-layer alignment pass after the onboarding series. Maintained
selection workflows now demonstrate the selection-only `PiPLSPathCV()` default, the generated fixed
reference includes `set_output()` and all public decomposition diagnostics, path details expose
`path_search_exhaustive_`, and guide-layer Pulp descriptions use `predictor_rank_profile()` instead
of direct candidate-column access.

Current status: **documentation aligned with the current implementation**.

### Guide-layer synchronization

A follow-up audit synchronizes the active `.llm` summaries with Decisions 0101--0105 and the
current repository. It replaces superseded example numbers, records the stable scorer-string
default, completes runtime and documentation-workflow ownership, and aligns the patch handoff with
the owner-requested downloadable patch, SHA-256 checksum, and concise five-command workflow. This
is a maintenance synchronization and introduces no new public or architectural decision.

Current status: **guide layer synchronized with the current repository**.

### One-standard-error visualization series

The owner-authorized series has two focused increments:

1. establish one public derived fold-based CV-MSE standard-error contract without changing plots;
2. migrate maintained CV-MSE figures and explanations while keeping the component choice manual.

Decision 0106 completes the first increment. It uses the conventional name one-standard-error rule
(1-SE rule), preserves the stored population fold SD, and derives the sample-standard-error-
equivalent quantity from the fold SD and split count. Automatic 1-SE selection remains out of scope.

Current status: **complete**. Maintained CV-MSE plots use the derived fold-based standard error,
the documentation explains the conventional 1-SE component heuristic, and component-count choice
remains explicit rather than automated.

### Component-path recommendation references

Decision 0107 adds exact stored-value minimum-CV-MSE and one-standard-error methods to
`PiPLSComponentPath`. The implementation returns complete immutable path rows without fitting,
refitting, mutation, comparison tolerances, or redundant state. A restrained follow-through
documents the methods in `docs/path_analysis.md` and the generated API route while keeping the
README, documentation home, tutorials, and renderers on explicit component choices. Decision 0108
then uses `one_standard_error_result()` in the advanced Tobacco workflow while retaining explicit
choices in the introductory and other numbered examples.

Current status: **implementation, reference documentation, and one focused Tobacco application
complete; not promoted in tutorials or entry pages**.

## Current next increment

Continue the owner-led human audit from the next clean focused snapshot. Do not prepare or publish a
package release.

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
8. return a downloadable root-relative patch and SHA-256 checksum, report validation, and provide
   the concise apply/check/stage/commit/snapshot command sequence requested by the owner.


## Public-result API cleanup

API1 is complete under Decision 0086. `PiPLSDecomposition` exposes descriptive final factors and
rank/solver diagnostics while the private core retains construction matrices. API2 is complete
under Decision 0087: scorer state is private, exact weight aliases are removed, and
`PiPLSPathCV` retains `cv_results_`, concise immutable path/profile results, standard `best_*`
attributes, `path_search_exhaustive_`, and `validation_report_` without execution-history or flat
OOF duplicates. Decision 0088 completes API3 by removing display-sign bookkeeping and redundant
synthetic zero blocks and by suppressing constructor signatures for returned immutable records.
