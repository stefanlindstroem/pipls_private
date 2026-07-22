# Decision 0075: two-tier tutorial route

## Status

Accepted and implemented.

## Context

Decision 0074 consolidated the normal user workflow into the Pulp tutorial and removed several
redundant guide pages. The resulting documentation is coherent, but the complete real-data tutorial
is too long to be the first encounter with path selection. A new programming user should be able to
learn the estimator and search roles before reading dataset provenance, OOF diagnostics, and the
full fitted-model plot catalogue.

The repository already has a deterministic synthetic train/test generator with known shared,
predictor-specific, and response-specific latent structure. That controlled setting can present the
selection contract with fewer variables and only three figures.

## Decision

Adopt a two-tier tutorial route:

1. `docs/tutorials/synthetic.md` is the first tutorial. It owns the shortest complete
   selection-and-prediction sequence: generate independent train/test data, evaluate
   `component_path_`, choose one component count, inspect its conditional predictor-rank profile,
   fit one fixed `PiPLSRegression`, and predict the external test block.
2. `docs/tutorials/pulp.md` is the second tutorial. It remains the complete real-data analysis with
   direct data loading, fixed-parameter OOF prediction, immutable inspection objects, and the
   distinction between standard PLS-family and Pi-PLS-specific plots.

Rename the former fixed synthetic example to `examples/02_synthetic_path_selection.py` and make it
the maintained source for the first tutorial. The example uses deterministic
`make_pipls_train_test()` data, selects two components and predictor rank four, and writes only:

```text
component_path.pdf
predictor_rank_profile.pdf
observed_vs_predicted.pdf
```

`tools/render_synthetic_tutorial.py` repeats the same small public-API calculation to generate three
single-chart SVG files plus a manifest. `make docs-figures` regenerates the synthetic assets before
the Pulp assets. Generated tutorial assets remain ignored build products.

The synthetic tutorial states that known generating ranks are interpretive references, not a
promise that cross-validation will recover them exactly. It ends after external-test prediction and
routes readers to the Pulp tutorial for OOF diagnostics and fitted-model interpretation.

## Consequences

The documentation now has a visible learning order from a short controlled workflow to a complete
real-data analysis. The home page and README route new users to the synthetic tutorial first. The
Pulp tutorial can later be shortened without creating an onboarding gap.

Decision 0074 remains authoritative for generated API, advanced-guide, and model-inspection
ownership. This decision supersedes only its assignment of the Pulp tutorial as the sole linear
description of normal use.
