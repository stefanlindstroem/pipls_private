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
- Predictor-rank modes: integer, `"max"`, and `"auto"`.
- General automatic rank bound uses the smallest training-fold size.
- Automatic selection uses fold-local response-standardized MSE.
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

Current status: **in progress**. This patch simplifies the local workflow by removing patch-
application and commit wrappers in favor of explicit Git commands. Phase A is complete when the
resulting tests and workflow checks pass and the change is committed.

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

### Phase B2: fixed-rank public estimator

Add `PiPLSRegression` for an explicit integer `predictor_rank` only.

Required outputs:

- scikit-learn constructor, validation, fitted-state behavior, `fit`, `predict`, `transform`, and
  scalar `score`;
- documented centering and scaling semantics;
- coefficient orientation and fitted attributes consistent with `.llm/public_api.md`;
- estimator-contract and pipeline smoke tests.

### Phase C1: rank-bound helper and `predictor_rank="max"`

Add one tested fold-safe rank-bound helper and the rule-fixed estimator mode. Do not add internal
cross-validation in the same increment.

### Phase C2: `predictor_rank="auto"`

Add conditional predictor-rank selection for one fixed `n_components`, including fold-local
preprocessing, response-standardized MSE, deterministic tie-breaking, diagnostics, and full-data
refitting.

### Phase D1: complete path analysis

Add `PiPLSPathCV` with the admissible triangular grid, shared materialized splits, standard
scikit-learn scorer orientation, and complete-pipeline cloning.

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

After this workflow patch is committed and a clean snapshot is produced, the next patch should be
**Phase B1: fixed-parameter private core**. It should use the supplied trusted implementations as
oracles and should remain limited to `_core.py`, focused tests, and the mathematical/numerical
contracts needed to describe that core.

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
