# Pi-PLS package development strategy

## Purpose

This file records current development principles, unresolved work, and the active review sequence.
Completed implementation history belongs in numbered decisions and Git history. `.llm/state.md`
contains the concise implemented-state handoff.

## Ownership

The project owner decides scientific scope, public API commitments, publication boundaries, and
roadmap order. The maintainer implements one bounded increment at a time, keeps source, tests,
documentation, decisions, and `.llm` synchronized, and returns a root-relative patch with complete
validation evidence.

## Development principles

- Preserve the mathematical construction in `.llm/mathematics.md` and `.llm/theory.md`.
- Keep the fixed estimator separate from cross-validated path search.
- Learn all model preprocessing inside each fit, including every training fold.
- Prefer explicit post-fit selection, refitting, OOF reporting, and inspection over hidden state.
- Keep public result objects immutable, validated, finite where required, and pickle-safe.
- Use scikit-learn conventions where they do not obscure Pi-PLS-specific semantics.
- Keep user data preparation visible; package-owned loaders are named reference conveniences, not a
  mandatory ingestion framework.
- Keep rendering caller-owned and numerical inspection package-owned.
- Avoid compatibility aliases during the unreleased pre-1.0 phase unless the owner explicitly
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

### Analysis

Pi-PLS-specific factorization inspection and shared PLS-family diagnostics are pure numerical
operations. Examples and users render the returned arrays directly. OOF diagnostics describe the
chosen selection under the search protocol and are not outer-validation estimates.

### Data

The package supports arbitrary user-provided arrays and three named immutable reference datasets.
Package resources are ordinary language-neutral files with one active matrix copy each. A future
dataset must have explicit redistribution and adaptation rights before inclusion.

### Product scope

This repository owns the installable package, user documentation, examples, tests, packaging, and
release validation. Paper reproduction and publication-only analyses remain downstream.

## Active sequence: Decision 0147

### Patch 1 -- decision lifecycle

Complete. Decision material is classified as current decisions, compact historical summaries, or
retired records recoverable from Git. Retirement is based on continuing relevance rather than age.

### Patch 2 -- compact maintainer context

Complete. Rewrite active `.llm` guidance around implemented contracts and unresolved work. Remove
completed patch narratives and stale intermediate API descriptions without changing executable or
served user behavior.

Acceptance conditions:

- no active guide claims that removed standard-error or transitional report APIs exist;
- `state.md` is a usable fresh-chat handoff;
- `strategy.md` contains current principles and roadmap rather than full increment history;
- `testing.md` describes current obligations rather than migration-by-migration fixtures;
- `analysis.md`, `public_api.md`, and `numerical_contracts.md` match the implementation;
- `.llm/decisions.md` remains structurally complete until decision retirement begins;
- the active guide layer is materially smaller without dropping durable contracts.

### Patch 3 -- retire explicitly superseded decisions

Next. Create a reviewed retirement map for decisions whose active contract is fully replaced by a
later canonical record. Remove or redirect every active reference in the same patch. Do not add
tombstone files or reuse decision numbers.

### Patch 4 -- historical summary and micro-decision consolidation

Add `docs/decisions/history.md` with compact development-era summaries and canonical links. Retire
one-off migration, naming, figure, documentation-arrangement, and completed cleanup decisions when
Git history or the summary is sufficient. The target of roughly 35--50 current decisions is a
review goal, not a test invariant.

### Patch 5 -- tests and snapshot policy

Reduce brittle AST/source-text assertions while retaining durable workflow and repository
boundaries. Split oversized repository-policy tests by responsibility where useful. Harden
snapshot creation so tracked caches, bytecode, generated documentation, coverage output, build
artifacts, and generated example outputs cause a clear failure, while ignored untracked local
artifacts remain excluded.

### Patch 6 -- dataset-module split

Move immutable dataset types, packaged-resource loading, and synthetic generation into private
modules. Keep `pipls.datasets` as the stable public facade. Require exact public imports, resource
bytes, deterministic generated arrays, validation, immutability, pickling, and distribution
contents.

### Patch 7 -- normalization and final audit

Normalize decision indexes and links; remove stale references; audit public/private imports,
removed names, generated artifacts, helper reachability, distribution contents, and `.llm`
duplication; then mark Decision 0147 implemented.

## Independent paused work

Decision 0139 Patch 3 may later refine presentation and navigation after an explicit owner request.
It is not part of Decision 0147 and must not be bundled into the cleanup sequence.

Future block-aware standardization remains intentionally undesigned. Do not reserve public names,
constructor parameters, or hidden abstractions before a dedicated owner-led design phase.

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
