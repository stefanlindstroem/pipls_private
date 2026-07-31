# Decision 0135: specific predictor-rank support warning name

## Status

Accepted and implemented.

## Context

The public warning class `StatisticalSupportWarning` is broader than the condition it reports.
Pi-PLS emits this warning only when the number of supplied observations is small relative to the
retained predictor rank: direct fixed fits warn below three observations per retained predictor
rank direction, while path-search support rules warn when configured below five supplied
observations per retained direction.

The package does not use this class for general statistical diagnostics, response support,
uncertainty estimates, or validation failures. The generic name therefore obscures the specific
contract and may imply a broader warning category than the implementation provides.

The package remains unreleased at version `0.0.0`, so no published compatibility commitment
requires retaining the former class name.

## Decision

Rename the public warning class to `PredictorRankSupportWarning`.

Use the new class throughout implementation, tests, pytest warning configuration, generated API
reference, troubleshooting guidance, and active guide-layer contracts. Do not add an alias,
fallback import, or serialization migration for `StatisticalSupportWarning`.

The warning thresholds, warning messages, stack levels, direct-fit behavior, and path-owned
suppression boundary remain unchanged. `PiPLSSearchCV` continues to suppress only this package
warning inside controlled feature probes, candidate fits, optional OOF fits, and the selected
full-data refit; unrelated warning categories remain visible.

Historical decision records remain unchanged. This decision supersedes Decisions 0009, 0031,
0032, and 0039 only where they require the former warning-class spelling. Their validation,
support-threshold, feasibility, and warning-ownership contracts otherwise remain in force.

## Consequences

- The public warning name states the exact support condition it represents.
- Direct fixed fits and path-search support rules retain their existing numerical thresholds.
- Warning filtering remains precise and package-specific.
- Existing pre-release code importing `StatisticalSupportWarning` must use
  `PredictorRankSupportWarning`.
- No duplicate compatibility warning class enlarges the public surface.
