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

No staged migration is active. Decision 0154's seven-patch predictor-rank migration is complete:
full-feasible automatic rank optimization uses exhaustive coverage by default, EPV is an explicit
fixed-rank policy, the pre-release `"max"`/`"rule"` shortcuts are absent from the active API, and
high-dimensional maintained examples request adaptive coverage explicitly where appropriate. The
release notes and clean installed-artifact smoke test cover the completed transition. Future changes
to predictor-rank selection require a new owner decision rather than extending this closed sequence.

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
