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

## Current model standardization and deferred block-aware variants

Model-internal centering and scaling are current, required estimator behavior; they are not deferred
product scope. `PiPLSRegression` mirrors the preprocessing contract of scikit-learn's
`PLSRegression`:

- every fit centers `X` and `Y` using statistics estimated from the data supplied to that fit;
- `scale=True` additionally divides both blocks by safe training-sample standard deviations;
- `scale=False` retains centering and uses unit scale vectors;
- every cross-validation candidate is a fresh estimator fit on one training fold, so validation
  observations never influence fold means or scales;
- after model selection, the chosen model is refitted and standardized on the complete training set
  supplied to `fit()`;
- prediction applies the stored training statistics and returns responses in their original units.

The fixed numerical core remains independent from preprocessing because it receives already
centered or centered-and-scaled matrices from the estimator layer. This separation does not make
standardization optional or external to model fitting.

Future development may add block-aware variants of model standardization. Their eventual public
placement—inside the estimator or as part of a supported model pipeline—is not designed yet. Only
the future API and block semantics are deferred. Regardless of placement, the complete model must
fit the scaling statistics inside each training fold and again during the final full-training
refit. They must not be fitted once to the complete dataset before cross-validation.

Until the project owner starts a dedicated design phase:

- no provisional public names, classes, constructor parameters, or block semantics are reserved;
- documentation must not imply that current centering/scaling is deferred;
- implementation work must not anticipate a future block-scaling API through hidden abstractions;
- ordinary model fitting and validation continue to use the implemented estimator-internal
  standardization contract.

A future block-aware standardization design requires a separate owner decision, explicit contracts,
and its own small reviewable increments.

## Current transition state

The package-product boundary and public navigation cleanup are complete. The repository no longer
contains paper-reproduction placeholders or promises to implement manuscript workflows.

The next product increment is to define the lightweight synthetic benchmark contract before
freezing benchmark results. That work does not include a block-aware standardization API design;
future block-aware variants remain deferred for months or until the project owner explicitly starts
a separate phase.
