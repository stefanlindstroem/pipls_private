# Pi-PLS package development strategy

## Purpose

This file records current development principles, unresolved work, and review protocol. Completed
implementation history belongs in numbered decisions and Git history. `.llm/state.md` contains the
concise implemented-state handoff.

## Ownership

The project owner decides scientific scope, public API commitments, publication boundaries, and
roadmap order. The maintainer implements one bounded increment at a time, keeps source, tests,
documentation, decisions, and `.llm` synchronized, and returns a root-relative patch with complete
validation evidence.

## Development principles

- Preserve the mathematical construction in `.llm/mathematics.md` and `.llm/theory.md`.
- Keep the fixed estimator separate from cross-validated path search.
- Learn all model preprocessing inside each fit, including every training fold.
- Prefer explicit post-search selection, immutable selection handoff, OOF reporting, refitting, and
  inspection over hidden state.
- Keep public result objects immutable, validated, finite where required, and pickle-safe.
- Use scikit-learn conventions where they do not obscure Pi-PLS-specific semantics.
- Keep user data preparation visible; package-owned loaders are named reference conveniences, not a
  mandatory ingestion framework.
- Keep rendering caller-owned and numerical inspection package-owned.
- Avoid compatibility aliases during the pre-1.0 phase unless the owner explicitly
  requests one.
- Use tests for durable behavior and machine contracts, not as a second copy of living prose.
- Keep patches small enough to review independently and do not combine future phases.

## Fixed architectural boundaries

### Modeling

`PiPLSRegression` fits one explicit rank pair. `PiPLSSearchCV` owns the triangular path, split
materialization, candidate evaluation, selection lookup, explicit full-data refit, and selection-
conditioned OOF reporting. Search fitting itself does not produce a final model.

### Selection

The final named rule vocabulary is `best_score` and tolerance-based `minimum_cv_mse`. Manual
selection is by evaluated component count. CV-MSE path summaries use equal weighting per
materialized split and population SD (`ddof=0`). There is no standard-error rule or result surface.

Decision 0148 adds a separate tolerance stage for conditional predictor-rank choice. Predictor-rank
tolerances are search-constructor controls because they determine `component_path_`; component-count
tolerances remain post-search `select()` and `refit()` controls. Exact numerical ties and adaptive
candidate coverage remain separate from substantive parsimony tolerances.

The implemented runtime now has `search_method="exhaustive"` as the default predictor-rank
coverage policy over the full hard-feasible domain. `search_method="adaptive"` remains an explicit
computational approximation, and `search_is_exhaustive_` continues to report whether the complete
candidate set was actually evaluated. Decision 0154 owns this boundary.

Decision 0143 implements `refit(selection=...)` as the exact-selection handoff for analytical
workflows. Rule-based and component-count refitting remain the compact route. Existing selections
use one compatibility definition across OOF reporting and final refitting.

Decision 0152 refines manual tutorial presentation. Inspect the unselected component path before
choosing a component count, treat setting that value and `search.select(...)` as one operation,
then review the selected path and conditional evidence with one possible return to selection.
Same-search OOF reporting is inspection rather than independent qualification.

### Analysis

Pi-PLS-specific factorization inspection and shared PLS-family diagnostics are pure numerical
operations. Examples and users render the returned arrays directly. OOF diagnostics describe the
chosen selection under the search protocol and are not outer-validation estimates. Fitted-value
diagnostics describe the refitted model on its training observations and remain explicitly
separate from predictive validation evidence.

### Data

The package supports arbitrary user-provided arrays and three named immutable reference datasets.
Package resources are ordinary language-neutral files with one active matrix copy each. A future
dataset must have explicit redistribution and adaptation rights before inclusion.

### Product scope

This repository owns the installable package, user documentation, examples, tests, packaging, and
release validation. Paper reproduction and publication-only analyses remain downstream.

## Current roadmap

Decision 0154's seven-patch predictor-rank migration is complete and closed. Future changes to
predictor-rank selection require a new owner decision rather than extending that sequence.

Decision 0155 opens an active six-step migration for response-subspace selection. The scientific and
API target is to retain the peer-reviewed cross-covariance construction as the default and add one
explicit least-squares/RRR-inspired software extension. The response-subspace policy belongs to the
fixed estimator and is propagated by search; it is not a third search dimension.

The six implementation steps are:

1. establish Decision 0155 and the maintainer contract;
2. implement both response-subspace policies in the fixed numerical core;
3. expose `response_subspace` on `PiPLSRegression` and preserve it through search, OOF work, and
   refitting;
4. add mathematical, numerical, and API regression coverage for both policies;
5. document the theory and API and add a programming-user comparison example;
6. complete the stale-contract, release, installed-artifact, and distribution audit.

Step 1 is split into four reviewable patches:

- 1A: record and index Decision 0155 -- complete;
- 1B: reconcile Decisions 0120, 0008, and 0039 with the accepted extension -- complete;
- 1C: open the active maintainer roadmap while leaving implemented contracts untouched -- complete;
- 1D: audit repository-wide consistency and close Step 1 before runtime implementation begins --
  complete.

The Step-1 audit confirmed that `.llm/public_api.md`, `.llm/mathematics.md`, `.llm/theory.md`,
`.llm/numerical_contracts.md`, source, tests, examples, and user-facing theory still describe only
the implemented cross-covariance runtime. No premature `response_subspace` public signature, test
expectation, or programming example is present. Those implemented-contract descriptions remain
unchanged until the corresponding source and test contracts land.

Step 2 is complete. It is split into four reviewable patches:

- 2A: isolate the current cross-covariance response-basis routine without changing numerics --
  complete;
- 2B: implement the isolated exact least-squares response-basis routine -- complete;
- 2C: add the two-value private core dispatch while preserving cross-covariance as the default --
  complete;
- 2D: harden numerical integration, synchronize implemented numerical contracts, and close Step 2 --
  complete.

Patch 2A changes only source organization around the already implemented response-basis SVD.
Patch 2B adds a private least-squares/RRR-inspired response-basis helper implemented by exact
reduced QR of `Z` followed by exact SVD of `Q_Z.T @ Y`; a projector test verifies equivalence to
Choice C on a well-conditioned problem. Patch 2C adds the private `fit_pipls_core()` dispatch with
exactly `"cross_covariance"` and `"least_squares"`; default and explicit cross-covariance fits are
covered for numerical identity, and the least-squares route uses the same downstream least-squares
map and `P`/`D`/`Q` diagonalization. Patch 2D adds full/seeded-randomized predictor-SVD coverage for
the least-squares route, protects the `p >> n` regime, and records that response-side QR/SVD remains
exact and introduces no independent rank threshold.

Step 3 is complete with three patches:

- 3A: expose and validate `response_subspace` on `PiPLSRegression`, preserve
  `"cross_covariance"` as the default, and forward the setting to the private core;
- 3B: certify direct-search, pipeline, OOF, and refit propagation without adding a third search
  dimension;
- 3C: synchronize the implemented public contract and close Step 3.

The existing estimator-template boundary required no search-runtime changes. Direct candidate
evaluation, explicit refit, selection-conditioned OOF reporting, and pipeline templates retain
`response_subspace="least_squares"` while search continues to overwrite only `n_components` and
`predictor_rank`. The ordinary `PiPLSSearchCV()` template still refits with
`response_subspace="cross_covariance"`.

Step 4 is complete with three reviewable patches:

- 4A: add independent mathematical reference coverage, RRR equivalence, least-squares training
  optimality, limiting-case identities, and shared core invariants -- complete;
- 4B: harden public scaling, serialization, configuration changes, and scikit-learn interoperability
  for the least-squares policy -- complete;
- 4C: synchronize internal mathematics/testing contracts and close Step 4 -- complete.

Patch 4A uses an independent Choice-C Gram/eigen calculation only in regression-test code; the
runtime continues to use the numerically preferred exact QR/SVD construction from Step 2. Patch 4B
adds public regression coverage for all scaling combinations, estimator/search serialization,
`set_params()` refitting, and external `GridSearchCV` while leaving runtime behavior unchanged.
Patch 4C records both response-subspace objectives, their shared downstream factorization, the RRR
and limiting-case identities, durable regression coverage, and the Decision-0154 full-domain rank
boundary in the authoritative internal contracts.

Step 5 is complete in three reviewable patches:

- 5A: rewrite the user-facing theory and establish the publication/software-extension boundary --
  complete;
- 5B: update API, workflow, reproducibility, performance, and manuscript-reproduction documentation
  -- complete;
- 5C: add the programming-user response-subspace comparison example and close Step 5 -- complete.

Patch 5A documents both response-subspace objectives, the exact QR/SVD realization of the
least-squares criterion, its RRR interpretation, the singular-value-weighting distinction between
the policies, their shared downstream least-squares/diagonalization stages, and the limiting cases
where their fitted maps coincide. Patch 5B propagates the implemented policy through the
fixed-regression and path-selection references, reproducibility and manuscript-alignment guidance,
computational-performance trade-offs, troubleshooting, and top-level discoverability. Patch 5C
adds the maintained matched-split Pulp comparison and labels its output as model-development CV
evidence rather than independent validation.

Step 6 is split into three reviewable patches:

- 6A: add release notes and reconcile stale active-contract/example-inventory wording -- complete;
- 6B: qualify both response-subspace policies in clean installed artifacts and execute Example 07
  from the source distribution -- complete;
- 6C: perform the final repository audit and close Decision 0155 -- next.

Patch 6A records the implemented feature under `Unreleased`, updates the current decision registry,
and synchronizes the maintained example inventory after Example 07. Historical records that were
correct when written remain historical rather than being rewritten. No runtime behavior changes in
6A. Patch 6B extends clean wheel/source-distribution smoke qualification to both response-subspace
policies, verifies least-squares policy propagation through installed search/refit, and executes both
Examples 01 and 07 from the extracted source distribution. No package runtime behavior changes in
6B.

The numerical compatibility invariant for Steps 2--6 is that the existing path and an explicit
`response_subspace="cross_covariance"` path reproduce the pre-Decision-0155 fixed-estimator
numerics, subject only to ordinary floating-point behavior. The least-squares route must remain
visibly outside the peer-reviewed companion publication throughout source, theory documentation,
and manuscript-reproduction guidance.

## Deferred work

The block-aware transformer itself remains outside this repository. `PiPLSRegression` now exposes
only the independent `scale_x` and `scale_y` controls needed for supported pipeline composition.
Do not reserve additional block-specific public names or hidden abstractions without a dedicated
owner decision.

## Maintenance protocol

For each patch:

1. start from the latest clean snapshot and record its source commit;
2. inspect affected source, tests, decisions, and contracts;
3. implement only the assigned increment;
4. run focused tests, the complete suite, compilation, formatting/diff audits, and applicable
   documentation/distribution checks;
5. compare numerics or bytes directly when behavior is supposed to remain unchanged;
6. independently apply the generated patch to a fresh extraction and repeat validation;
7. provide the patch, checksum, concise behavior summary, and routine owner commands.

When a tool such as Ruff, mypy, MkDocs, or an optional renderer dependency is unavailable, state
that precisely and provide the authoritative local command. Do not claim an unrun check passed.
