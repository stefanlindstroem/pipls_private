# Package product scope

## Purpose

The `pipls` repository is the long-lived home of the installable Pi-PLS software product. It is not
the reproduction repository for any one scientific paper. Publications may motivate or validate
features, but they consume released versions of `pipls` rather than defining this repository's
layout or roadmap.

This boundary lets the package evolve across multiple publications while keeping its public surface
focused on programming users.

## Repository responsibilities

The repository owns:

- the installable `pipls` package and its supported public API;
- numerical, estimator, model-selection, validation, and synthetic-data functionality;
- user documentation and API reference;
- concise executable examples that demonstrate ordinary package use;
- transparent redistributable reference datasets;
- lightweight deterministic validation benchmarks;
- package tests, compatibility policy, release notes, packaging, and versioned releases.

These assets should answer practical software-user questions: how to install the package, construct
`X` and `Y`, fit and validate Pi-PLS models, interpret outputs, compose supported pipelines, and
understand numerical or performance trade-offs.

## Publication boundary

Paper-specific reproduction assets do not belong in this repository:

- manuscript figure and table generation;
- complete publication simulation grids;
- paper-only OLS, CCA, or other comparator implementations;
- cached manuscript results;
- manuscript-specific reporting code;
- publication-specific pinned environments;
- orchestration whose purpose is to reproduce one paper rather than validate the package.

A paper-reproduction repository should depend on a tagged `pipls` release and may contain those
assets independently. The `pipls` repository may link to such repositories after they become
public, but it does not absorb their workflows.

Ordinary PLS is a relevant lightweight package benchmark because it is the nearest practical
baseline for Pi-PLS users. OLS or CCA belong here only when they protect a package-level
mathematical identity, limiting case, or user-facing behavior; paper-only comparisons remain
downstream.

## Benchmark boundary

Package benchmarks are lightweight validation assets, not publication claims. Synthetic benchmarks
should be primary because ranks, latent structure, signal strengths, noise, and subspaces are known.
Reference real datasets provide representative smoke checks and usage validation.

Benchmarks should protect such questions as:

- recovery of known synthetic structure;
- prediction and rank-selection behavior;
- consistency between exact and randomized numerical paths;
- comparison with ordinary PLS under controlled conditions;
- runtime and memory regressions relevant to package users.

Large experiment grids, final paper numbers, and manuscript plotting stay outside this repository.

## Future preprocessing

Future package development may add general preprocessing capabilities, including standardization
pipelines and block-scaling functionality. That direction remains valid, but no preprocessing or
block-scaling API is designed, scheduled, or implied by the current roadmap.

Until the project owner starts a dedicated design phase:

- the fixed Pi-PLS numerical core remains independent from preprocessing;
- learned preprocessing remains fold-local when composed around supported estimators;
- no provisional public names, classes, constructor parameters, or block semantics are reserved;
- documentation should describe the future area only as deferred product scope;
- implementation work must not anticipate the future API through hidden abstractions.

A future preprocessing design requires a separate owner decision, explicit contracts, and its own
small reviewable increments.

## Near-term transition

The current repository still contains paper-oriented placeholders and language inherited from its
initial publication plan. The accepted near-term sequence is:

1. update the `.llm` guide layer to establish this package-product boundary;
2. remove publication placeholders and rewrite public repository navigation around the software
   product;
3. define the lightweight synthetic benchmark contract before freezing benchmark results.

The third step does not include a preprocessing or block-scaling API design. That work remains
deferred for months or until the project owner explicitly starts it.
