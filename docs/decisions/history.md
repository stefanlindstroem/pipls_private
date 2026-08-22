# Decision history

This page summarizes completed development eras whose intermediate numbered decisions no longer
belong in the current policy set. It is not an exhaustive patch log and does not replace Git
history. Current behavior is defined by the retained numbered decisions, the implementation, and
the active maintainer contracts.

## Numerical and search foundation

The earliest work established the fixed Pi-PLS construction, fold-local response-standardized loss,
rank feasibility, exhaustive and adaptive predictor-rank search, and scikit-learn-compatible path
evaluation. Intermediate path classes, shared-engine arrangements, default-support experiments, and
API-polish passes were later replaced by the fixed-estimator/search boundary and explicit post-fit
selection lifecycle.

Current canonical records: [0001](0001-core-definition.md),
[0002](0002-preprocessing-semantics.md), [0003](0003-predictor-rank-selection.md),
[0004](0004-response-standardized-mse.md), [0007](0007-predictor-rank-search-policies.md),
[0008](0008-predictor-svd-policy.md), [0009](0009-public-parameter-validation.md),
[0014](0014-validation-metadata-scope.md), [0032](0032-full-sample-rank-support.md),
[0039](0039-fixed-estimator-path-search-boundary.md),
[0092](0092-fold-numerical-rank-feasibility.md), [0102](0102-path-search-defaults.md),
[0137](0137-post-fit-inspect-decide-refit-lifecycle.md),
[0140](0140-search-owned-path-selection.md),
[0143](0143-model-selection-provenance-and-oof-reporting.md),
[0146](0146-cv-mse-tolerance-selection.md),
[0154](0154-full-domain-predictor-rank-selection.md), and
[0155](0155-response-subspace-selection-policies.md).

## Datasets and product boundary

The repository moved from repository-relative example matrices and candidate integrations to a
closed, package-owned set of Pulp, Sugarcane, and Tobacco resources. Linnerud was removed, and
unlicensed legacy candidates remain excluded. Synthetic generators remain local, deterministic,
and independent of a generic registry or downloader. Publication reproduction remains downstream
of the package repository.

Current canonical records: [0015](0015-dataset-and-synthetic-api.md),
[0024](0024-package-product-repository-boundary.md),
[0025](0025-model-internal-standardization-boundary.md),
[0041](0041-legacy-dataset-licensing-roadmap.md),
[0119](0119-manuscript-latent-geometry-generator.md),
[0123](0123-companion-manuscript-synthetic-data-guide.md), and
[0142](0142-package-owned-reference-datasets.md).

## Examples and documentation

The example series evolved through several numbering, ownership, and tutorial arrangements. The
maintained outcome is a self-contained six-example route led by an automatic Pulp fit, followed by
manual selection review, one unified PLS-family path comparison, and complete real-data
interpretation workflows. The unified comparison overlays the publication-default and least-squares
Pi-PLS response policies with ordinary PLS on matched folds for all three reference datasets plus
one fixed near-saturated synthetic stress case. The Home page derives a simplified Pulp/Tobacco
view from that same numerical protocol to illustrate possible parsimony in shared component count
without exposing the optional least-squares policy in the introductory motivation. Served
documentation is self-contained, separates tutorials, guides, and generated reference, and keeps
maintainer decisions outside the user site. Semantic cross-referencing now gives reference datasets,
the companion publication, and high-value mathematical concepts stable destinations across those
layers without mechanically linking every repeated occurrence.

Current canonical records: [0065](0065-documentation-layer-consolidation.md),
[0152](0152-selection-review-feedback-workflow.md),
[0155](0155-response-subspace-selection-policies.md),
[0156](0156-unified-pls-family-path-comparison.md),
[0157](0157-near-saturated-synthetic-pls-comparison.md),
[0158](0158-home-page-parsimony-comparison.md),
[0159](0159-documentation-cross-reference-architecture.md), and the product and workflow
boundaries in
[0024](0024-package-product-repository-boundary.md),
[0045](0045-pls-family-analysis-boundary.md), and
[0142](0142-package-owned-reference-datasets.md).

## Inspection and rendering

Inspection developed from package-owned plotters and report objects into validated immutable
numerical results with caller-owned rendering. Intermediate atomic plotters and direct-rendering
migration records are historical. The package owns coordinates, diagnostics, and orientation; users
and examples own Matplotlib artists, panels, labels, layout, and file output.

Current canonical records: [0042](0042-model-inspection-and-post-analysis.md),
[0045](0045-pls-family-analysis-boundary.md),
[0061](0061-example-owned-report-composition.md),
[0083](0083-data-first-rendering-policy.md), [0094](0094-inspection-result-safety.md),
[0110](0110-response-anchored-display-factors.md),
[0124](0124-mathematical-typography-and-subscripts.md),
[0141](0141-spectral-predictor-rank-profile-figures.md), and
[0160](0160-graceful-adjusttext-fallback.md).

## Packaging and release engineering

Documentation builds, compatibility matrices, clean wheel and source-distribution tests, optional
dependency groups, generated tutorial assets, snapshot creation, and deployment were introduced in
separate increments. Their durable outcome is a strict self-contained documentation build, tested
Python and dependency ranges, isolated artifact validation, optional graphics dependencies, and
clean committed-tree snapshots.

Current canonical records: [0054](0054-compatibility-policy.md),
[0065](0065-documentation-layer-consolidation.md),
[0091](0091-clean-git-snapshots.md),
[0103](0103-installation-and-optional-dependency-boundary.md),
[0117](0117-commercial-license-authorship-and-citation.md), and
[0147](0147-decision-lifecycle-and-maintainer-context.md).

## Pre-release API normalization

Before the first release, repeated naming and cleanup passes removed aliases, duplicated fitted
state, shape-only properties, redundant result access, public-looking private modules, and unused
helpers. Those intermediate records are historical because no compatibility period was required at
version `0.0.0`. The current names and export boundaries are defined by the mathematical terminology,
result invariants, focused modules, and final implementation-surface decision.

Current canonical records: [0039](0039-fixed-estimator-path-search-boundary.md),
[0093](0093-public-result-invariants.md),
[0120](0120-companion-manuscript-theory-alignment.md),
[0121](0121-canonical-pipls-terminology.md), and the current API decisions listed above.

## Completed workflow, terminology, and cleanup sequences

Several late pre-release records described bounded migrations rather than independent durable
contracts. The three-stage onboarding sequence was completed and refined into the current
selection-review workflow. The search-policy rename settled on `"adaptive"` and `"exhaustive"`, the
computational-performance guide became ordinary served documentation, exact selection handoff was
folded into the generic provenance and OOF contract, package-owned leave-one-out support was removed,
and the behavioral test-suite cleanup replaced source and prose policing with executable boundaries.

Current canonical records: [0007](0007-predictor-rank-search-policies.md),
[0065](0065-documentation-layer-consolidation.md),
[0083](0083-data-first-rendering-policy.md),
[0143](0143-model-selection-provenance-and-oof-reporting.md),
[0147](0147-decision-lifecycle-and-maintainer-context.md), and
[0152](0152-selection-review-feedback-workflow.md). The served
[computational-performance guide](../computational_performance.md) records the current user-facing
performance guidance.

## Retired experiments and one-off cleanups

The removed benchmark layer, abandoned candidate datasets, temporary aliases, one-time navigation
changes, and patch-specific cleanup records have no remaining independent contract. Git retains the
full documents. The current retirement policy and exhaustive replacement map are defined by
[0147](0147-decision-lifecycle-and-maintainer-context.md) and
[retirements.md](retirements.md).
