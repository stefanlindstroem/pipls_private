# Pi-PLS package development strategy

## Purpose

This file records current development principles, unresolved work, and review protocol. Completed
implementation history belongs in numbered decisions, `docs/decisions/history.md`, and Git history.
`.llm/state.md` contains the concise implemented-state handoff.

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
- Keep user data preparation visible; reference loaders are conveniences, not an ingestion layer.
- Keep rendering caller-owned and numerical inspection package-owned.
- Avoid compatibility aliases during the pre-1.0 phase unless the owner explicitly requests one.
- Use tests for durable behavior and machine contracts, not as a second copy of living prose.
- Keep patches small enough to review independently and do not combine future phases.

## Fixed architectural boundaries

### Modeling

`PiPLSRegression` fits one explicit rank pair. `response_subspace` is fixed-estimator configuration:
`"cross_covariance"` is the peer-reviewed default and `"least_squares"` is a software extension.
`PiPLSSearchCV` preserves that template policy and searches only component count and predictor rank.

### Selection

`PiPLSSearchCV.fit()` owns split materialization and path evaluation but does not produce a final
model. Exhaustive coverage of the full hard-feasible optimized predictor-rank domain is the default;
adaptive coverage is an explicit approximation and EPV is an explicit fixed-rank policy.

`search.select()` is the sole public selected-row lookup. The named component-count rules are
`best_score` and tolerance-based `minimum_cv_mse`; manual selection is by evaluated component count.
Predictor-rank tolerances are search-constructor controls because they determine the conditioned
component path, while component-count tolerances are post-search selection controls. Exact numerical
ties remain separate from substantive parsimony tolerances.

Decision 0143 owns exact selection handoff across `oof_report(selection=...)` and
`refit(selection=...)`. Decision 0165 refines the maintained workflow: component-path evidence and,
when useful, the conditional predictor-rank profile complete ordinary selection before OOF
diagnostics are inspected. Same-search OOF diagnostics are selection-conditioned inspection rather than
independent post-selection qualification or validation, and using them to retune the model makes
them additional selection evidence that requires an appropriate outer assessment for performance claims.

### Analysis

Pi-PLS-specific factorization inspection and shared PLS-family diagnostics are pure numerical
operations. Examples and users render returned arrays directly. OOF diagnostics describe the chosen
selection under the search protocol; fitted-value diagnostics describe the refitted model on its
training observations and remain explicitly separate.

### Data

The package supports arbitrary user-provided arrays and three named reference datasets with
read-only model arrays.
Package resources are ordinary language-neutral files with one active matrix copy each. A future
dataset requires explicit redistribution and adaptation rights before inclusion.

### Product scope

This repository owns the installable package, user documentation, examples, tests, packaging, and
release validation. Paper reproduction and publication-only analyses remain downstream.

## Current roadmap

Decision 0147 is complete. Decision-registry consolidation, active `.llm` compaction, stale-test
pruning, and validation-ownership cleanup are finished. Ordinary pytest is behavior-focused;
complete examples and tutorial rendering are owned by their dedicated Make targets and CI jobs.

Decision 0164 is complete. Patches 0164A--0164F replaced the overgrown Reference with a lean flat
lookup surface, slimmed both estimator pages, merged inspection concepts and API, retained the
synthetic-generator structural explanation and latent-role figure, and finished the overview,
datasets/generators, Troubleshooting, and cross-reference audit. No runtime, numerical, public-API,
dataset, or generator behavior changed.

Decision 0165 is complete. Patches 0165A--0165D establish that path evidence selects the model
while OOF diagnostics characterize the accepted selection, align all three maintained tutorials
with that workflow, split the Reference into Path and selection plus OOF diagnostics, and complete
the terminology and cross-reference audit. This sequence changes documentation methodology and
ownership only.

## Deferred work

The block-aware transformer itself remains outside this repository. `PiPLSRegression` exposes only
the independent `scale_x` and `scale_y` controls needed for supported pipeline composition. Do not
reserve additional block-specific public names or hidden abstractions without a dedicated owner
decision.

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
