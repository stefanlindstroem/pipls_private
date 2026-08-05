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

Decision 0148 adds a separate tolerance stage for conditional predictor-rank choice. Predictor-rank
tolerances are search-constructor controls because they determine `component_path_`; component-count
tolerances remain post-fit `select()` and `refit()` controls. Exact numerical ties and adaptive
candidate coverage remain separate from substantive parsimony tolerances.

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

## Current roadmap

Decision 0148 governs the active five-patch predictor-rank tolerance increment:

1. establish the decision and accepted contracts -- complete;
2. separate exact numerical score comparison from substantive rank-tolerance qualification --
   complete;
3. implement constructor controls, hierarchical path selection, public evidence, and API tests --
   complete;
4. demonstrate separate 10% predictor-rank and component-count tolerances in Tobacco;
5. complete migration, documentation, distribution, and repository audits.

Patch 4 is next. It must apply a separately named 10% predictor-rank relative tolerance on the
Tobacco search constructor while retaining the existing separately named 10% component-count
relative tolerance on `refit()`. The workflow must report and render both reference and retained
choices without conflating their thresholds.

## Independent paused work

Decision 0139 Patch 3 may later refine presentation and navigation after an explicit owner request.
It is not part of Decision 0148 and must not be bundled into the predictor-rank tolerance sequence.

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
