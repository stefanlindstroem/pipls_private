# Decision 0051: core generated API reference

## Status

Accepted and implemented.

## Context

The strict MkDocs foundation built the handwritten guides, but the primary estimator signatures,
fitted attributes, shapes, and conditional output contracts were duplicated across prose and source.
The core public objects require one generated reference derived from their Python docstrings.

## Decision

- Add mkdocstrings-python and Ruff to the dedicated documentation dependency group; Ruff formats
  separated generated signatures in a docs-only environment.
- Resolve the package from `src/` and parse the existing NumPy-style docstrings.
- Generate explicit pages for `PiPLSRegression`, `PiPLSPathCV`, `PiPLSDecomposition`,
  `PiPLSValidationReport`, and `StatisticalSupportWarning`.
- List class members explicitly so inherited implementation machinery and private modules are not
  exposed accidentally.
- Document constructor parameters, fitted attributes, array shapes, return values, and conditional
  `refit` and OOF behavior in the public source docstrings.
- Require the generated core pages to cover every supported top-level object except `__version__`.
- Keep inspection, plotting, datasets, and metrics for the next generated-reference increment.

## Consequences

`make docs` now verifies that core public objects resolve through mkdocstrings and that their
docstrings remain renderable. The generated reference uses supported public import paths, while
private modules and inherited implementation details remain outside the site. Runtime dependencies
and numerical behavior are unchanged.
