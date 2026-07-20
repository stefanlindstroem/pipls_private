# Decision 0046: keep numbered examples concise and trust repository-owned inputs

## Status

Accepted and implemented.

## Context

The numbered examples had accumulated application-style scaffolding: one-use configuration
constants, repeated path reporting, checks of committed CSV headers, dtype conversions for headers
already read as strings by pandas, and directory creation before every write. Those details obscured
the estimator, selection, inspection, and reporting calls that the examples are intended to teach.

The repository controls the committed datasets and the output layout under `examples/results/`.
Their structure is tested separately. A numbered example therefore does not need to defend against
repository corruption as though it were an independent production application.

## Decision

Keep the scientific stages visible, but remove scaffolding that does not help explain them.

- Inline arguments and labels that are used only once when the function signature already explains
  their meaning.
- Retain named constants only for genuine user choices or values reused across a workflow.
- Read committed `X.csv` and `Y.csv` directly and use their headers without redundant conversions or
  order assertions.
- Assume the tracked output directories exist. Numbered examples and their support functions do not
  create directories.
- Track `examples/results/` and the three post-analysis subdirectories with `.gitkeep` files while
  continuing to ignore generated CSV and PDF artifacts.
- Preserve those tracked directories during `make clean`.
- Keep validation of reusable numerical and artifact contracts in package code and focused tests;
  do not repeat those checks in numbered scripts.
- Retain concise completion output rather than listing every generated file.

The complete workflows remain explicit about their three scientific stages: component-path
comparison, fixed Pi-PLS fitting, and Pi-PLS post-analysis.

## Consequences

The numbered examples are shorter and place the calls a programming user should adapt near the
center of attention. Missing or damaged repository files now fail naturally at the read or write
operation. The reusable helper contracts and their focused tests remain unchanged, except that
helpers also assume their caller supplies an existing output directory.

This decision does not change fitted models, selected parameters, CV splits, numerical artifacts,
plot contents, or public package APIs.
