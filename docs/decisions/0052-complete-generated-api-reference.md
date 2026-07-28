# Decision 0052: complete generated API reference

## Status

Accepted and implemented.

## Context

The core generated reference covered the fixed estimator, search meta-estimator, decomposition, validation
report, and warning. The supported inspection, plotting, dataset, and metric submodules still relied
on handwritten guide prose and sparse source docstrings.

## Decision

- Generate explicit API pages for `pipls.inspection`, `pipls.plotting`, `pipls.datasets`, and
  `pipls.metrics`.
- Cover every name declared public by those submodules without expanding complete modules or
  exposing private helpers.
- Keep Pi-PLS-specific factorization inspection distinct from estimator-neutral PLS-family
  analysis.
- Document Matplotlib as optional and require importing `pipls.plotting` to work when Matplotlib is
  unavailable; plotting functions import it only when called.
- Document immutable result fields, array shapes, zero-based selections, prediction provenance,
  synthetic latent structure, and scorer sign conventions in the public source docstrings.
- Define the four supported dataset objects through `pipls.datasets.__all__`.
- Continue to exclude `pipls.model_selection` and private modules from the generated reference.

## Consequences

The generated site now covers the complete supported public API. Documentation tests compare each
published submodule page with its declared `__all__`, so additions or removals require an explicit
reference update. Runtime dependencies and numerical behavior are unchanged; the dataset `__all__`
only formalizes the existing supported import surface.
