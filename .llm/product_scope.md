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
- numbered user examples ranging from one minimal fit to complete reference workflows;
- transparent redistributable reference datasets;
- package tests, compatibility policy, release notes, packaging, and versioned releases.

These assets should answer practical software-user questions: how to install the package, construct
`X` and `Y`, fit and validate Pi-PLS models, interpret outputs, compose supported pipelines, and
understand numerical or performance trade-offs.

## Human and machine usability

Every package-facing artifact should be comfortable for both human readers and automated tools.
Prefer familiar, inspectable formats with explicit machine contracts: CSV for genuinely tabular
results, YAML for concise structured metadata, and ordinary Python interfaces for computation.
Avoid nesting, opaque encodings, or infrastructure-oriented formats when a flat representation is
more natural.

Machine readability does not justify making routine inspection difficult. Human readability does
not justify ambiguous columns or undocumented types. Versioned schemas, stable names, explicit null
semantics, and straightforward examples should serve both audiences together.

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

Ordinary PLS remains relevant in the dedicated component-path comparison example. Broader
comparative studies, including synthetic recovery, nuisance sensitivity, solver comparisons,
runtime, and memory studies, belong in downstream research or reproduction repositories unless a
future package contract requires a focused regression test.

## Current model standardization and external block-aware variants

Model-internal centering and scaling are current estimator behavior; they are not deferred product
scope. `PiPLSRegression` keeps `scale` as the backward-compatible policy for both blocks and also
permits independent predictor and response overrides:

- every fit centers `X` and `Y` using statistics estimated from the data supplied to that fit;
- `scale_x=None` and `scale_y=None` inherit `scale`;
- an enabled block is divided by safe training-sample standard deviations after centering;
- a disabled block remains centered and uses a unit scale vector;
- every cross-validation candidate is a fresh estimator fit on one training fold, so validation
  observations never influence fold means or scales;
- after model selection, the chosen model is refitted and standardized on the complete training set
  supplied to `fit()`;
- prediction applies the stored training statistics and returns responses in their original units.

The fixed numerical core remains independent from preprocessing because it receives already
centered or centered-and-scaled matrices from the estimator layer. This separation does not make
standardization optional or external to model fitting.

Block-aware predictor standardization belongs outside `PiPLSRegression`, for example in a dedicated
transformer repository composed through the supported scikit-learn pipeline boundary. The terminal
Pi-PLS estimator can use `scale_x=False` so upstream predictor scaling is preserved while choosing
response scaling independently with `scale_y`. Block definitions, block norms, and block-scaling
methods remain outside this repository. Any learned transformer must be fitted inside each training
fold and again during the final full-training refit, never once on the complete dataset before
cross-validation.

For block-aware work in `pipls`:

- `scale_x` and `scale_y` are the complete estimator-side composition controls currently accepted;
- no block-aware class, block definition, scaling method, or hidden block abstraction belongs in
  this repository without a separate owner decision;
- documentation must distinguish estimator scaling controls from the external transformer's block
  semantics;
- ordinary model fitting and validation continue to use the implemented fold-local standardization
  boundary.

The external block-aware transformer design requires its own explicit contracts and reviewable
increments in the downstream repository.

## Current transition state

The package-product boundary and public navigation cleanup are complete. The repository no longer
contains paper-reproduction placeholders or promises to implement manuscript workflows.

The former synthetic benchmark sequence was retired after serving its development-validation
purpose. Package-owned Pulp, Sugarcane, and Tobacco are user-run examples rather than default-test
jobs. Every maintained consumer uses the installed loaders, and the package resources are the sole
active matrix copies. Example 03 keeps the Pi-PLS and
standard PLS (NIPALS) comparison paths in memory
and writes only the overlaid final PDFs. Sugarcane keeps its Pi-PLS path, selection-conditioned OOF
predictions, and inspection results in memory and writes only final PDF figures. Pulp follows the
same in-memory boundary through `load_pulp()` and adds separate unselected and selected
component-path figures plus a conditional predictor-rank-profile figure.
Tobacco is also direct while preserving full-SVD spectral analysis, response pagination, and
observation diagnostics. Every analytical model-producing workflow creates one visible selection
before optional OOF inspection and final fixed-model fitting. The package now provides only the
independent predictor/response scaling controls needed for external block-aware pipeline
composition; block-aware transformer design itself remains downstream.
