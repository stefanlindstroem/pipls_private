# Pi-PLS implementation strategy

## Status and purpose

This is the operational implementation strategy for the repository. It converts the detailed
publication plan in `docs/publication_repository_plan.md` into small, testable increments that
can be maintained during LLM-assisted development.

The detailed publication plan records the complete intended architecture and scientific
rationale. This file records the current phase, the next admissible increment, and the
maintenance rules used for day-to-day development. When the two documents disagree, stop and
resolve the discrepancy explicitly rather than silently choosing one.

## Ownership

- The project owner decides scientific aims, publication policy, public API commitments, and
  unresolved methodological choices.
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
4. Add focused tests with every behavioral change.
5. Compare subspaces, regression maps, and predictions rather than raw singular-vector signs.
6. Fit every learned preprocessing operation inside the corresponding training fold.
7. Keep paper-specific orchestration explicit and separate from general estimator defaults.
8. Update the relevant `.llm` contract and user-facing documentation when a contract changes.
9. Return one root-relative Git patch per increment with an explicit validation report.
10. Use direct `git apply`, `git add`, and `git commit` commands; do not maintain wrapper scripts
    for patch application or committing.
11. Do not combine algorithm porting, API expansion, dataset migration, and paper reproduction in
    one patch unless the dependency cannot be separated.

## Fixed architectural decisions

- Runtime package layout: `src/pipls/`.
- Public estimator style: scikit-learn compatible.
- Public names: `n_components`, `predictor_rank`, and
  `samples_per_predictor_rank`; no public aliases `h`, `r_pi`, or `c`.
- Standard response subspace for the paper release: leading right singular vectors of
  `Z.T @ Y`.
- Predictor-rank modes target four semantics: integer, `"max"`, `"optimal"`, and `"auto"`.
- `"optimal"` means exhaustive CV evaluation of every admissible predictor rank.
- `"auto"` means deterministic adaptive coarse-to-fine search and is explicitly approximate.
- General search bounds use the smallest training-fold size.
- All CV-based selection uses fold-local response-standardized MSE.
- Search approximation and linear-algebra approximation are separate policies. Randomized SVD
  must be introduced through an explicit solver contract and diagnostics, not hidden inside the
  meaning of `predictor_rank`.
- Custom learned preprocessing is searched around the complete pipeline.
- Paper-specific rank rules and LOO reporting conventions are explicit reproduction inputs.
- `.llm/` is tracked repository infrastructure and is excluded from the installable package.
- Patch application and commits use ordinary Git commands rather than project wrapper scripts.
- Snapshots contain repository-root contents without an enclosing project directory.
- Patches are unified Git patches relative to repository root.

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

### Phase D2: LOO and advanced split protocols

Add ordered out-of-fold predictions, the documented LOO protocol, grouped and temporal examples,
and explicit selection-conditioned reporting.

### Phase E: datasets and synthetic generator

Add the common dataset schema, generic loader, deterministic converters, provenance and licensing
checks, and the side-effect-free synthetic generator. Migrate one dataset per increment where
practical.

### Phase F: paper reproduction and release

Add paper scripts, frozen result tolerances, build/install smoke tests, release metadata, DOI
workflow, and a clean tagged paper release.

## Current next increment

The next implementation patch should be **Phase D1: complete path analysis**. It should add
`PiPLSPathCV` over the admissible triangular $(h,r_\pi)$ surface, reuse one materialized split set,
fit preprocessing inside each training fold, mirror the established `"optimal"` and `"auto"`
search vocabulary, and expose a refitted best estimator plus reconstructable path diagnostics.

## Maintenance protocol

For every patch, the LLM maintainer should:

1. read `.llm/README.md`, this file, and the relevant contracts;
2. identify the current phase and avoid work assigned to later phases;
3. state which acceptance conditions the patch addresses;
4. update `Current status` and `Current next increment` when phase state changes;
5. update fixed decisions only after an explicit owner decision;
6. run and report each applicable Makefile validation target;
7. return a root-relative patch and provide the exact direct Git commands for checking, applying,
   inspecting, staging, and committing it.
