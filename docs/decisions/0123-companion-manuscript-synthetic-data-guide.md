# Decision 0123: companion-manuscript synthetic-data guide

## Status

Accepted and implemented.

## Context

Decision 0119 added `make_pipls_latent_geometry()` as the exact Gaussian latent-data generator used
by the companion manuscript. Decisions 0120--0122 aligned the canonical theory and terminology.
The public documentation still lacked one focused route explaining what the generator reproduces,
how a fixed seed identifies one realization, and why complete manuscript tables and figures require
additional publication-specific assets.

The package repository must expose enough information to use the manuscript data-generating model
without absorbing complete simulation grids, comparator pipelines, cached results, or figure and
table orchestration. Its practical real-data workflow and general search policy must remain
unchanged.

## Decision

Add a served companion-manuscript synthetic-data guide that distinguishes three targets:

1. reproducing the Gaussian data-generating distribution;
2. reproducing one deterministic seeded dataset;
3. reproducing complete manuscript tables or figures.

The guide must:

- state the two latent-geometry equations and the independent Gaussian draw assumptions;
- demonstrate `make_pipls_latent_geometry()` with a representative manuscript configuration;
- show direct verification of the stored score, loading, signal, and noise arrays;
- explain which version, seed, dimensions, noise settings, and numerical-environment details must
  be recorded for a realization-level claim;
- document the oracle synthetic dimensions $r_\pi=d_p+d_s$ and $h=d_s$ without presenting them as
  a real-data selection rule;
- state that complete publication reproduction additionally requires parameter grids, all seeds,
  preprocessing, validation, comparators, aggregation, dependency pins, and reporting code;
- route complete publication orchestration to a downstream repository that pins a tagged `pipls`
  release;
- state explicitly that `PiPLSSearchCV`, its defaults, and maintained real-data workflows are not
  changed or replaced.

Expose the guide through served navigation and cross-link it from the theory, dataset API,
reproducibility, citation, home, and README routes. Do not add a manuscript script, benchmark,
example, cached result, or generated figure.

## Consequences

- Users can identify precisely which level of manuscript reproducibility the package supports.
- The exact synthetic distribution is documented with executable public API calls and truth
  identities.
- A seed is not presented as sufficient without its complete generation context.
- The repository remains a general software product rather than a publication-reproduction
  pipeline.
- No production code, estimator behavior, search policy, validation protocol, benchmark, example,
  rendering, or real-data workflow changes.
