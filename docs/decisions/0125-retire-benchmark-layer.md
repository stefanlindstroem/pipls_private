# Decision 0125: Retire the benchmark layer

## Status

Accepted.

## Context

The repository contained four focused synthetic benchmark scripts for fixed-structure recovery,
rank selection, comparison with ordinary PLS under predictor-specific nuisance, and full-versus-
randomized solver consistency. Each script had dedicated plumbing and output-contract tests, a
public documentation page, and a normative maintenance contract.

These assets were useful while the implementation and numerical policies were being established.
They did not define package acceptance thresholds, and their maintained numerical properties are
now covered more directly by focused numerical, estimator, search, synthetic-data, and solver tests.
Keeping the benchmark layer therefore adds a separate repository subsystem without protecting an
independent public contract.

## Decision

Remove the complete active benchmark layer:

- delete `benchmarks/` and `tests/benchmarks/`;
- delete `docs/benchmarks.md` and `.llm/benchmarking.md`;
- remove benchmark-specific result paths, packaging rules, lint and formatting inputs, cleanup
  rules, navigation, contributor instructions, and repository tests;
- remove active benchmark ownership from current product, project, testing, development, state, and
  strategy contracts.

Historical decision records remain unchanged. They continue to explain why the benchmark subsystem
was introduced and how it evolved. This decision supersedes their active benchmark requirements.

Do not migrate descriptive benchmark calculations into ordinary tests merely to retain them. A
numerical property belongs in the test suite only when it is a maintained package contract.
Comparative scientific studies and performance investigations belong in downstream research or
reproduction repositories unless a future decision establishes a new focused package contract.

## Consequences

- The repository has no benchmark directory, benchmark command, generated benchmark result, or
  benchmark-specific public documentation.
- `make check`, documentation builds, source distributions, and snapshots no longer carry benchmark
  infrastructure.
- Estimator, search, synthetic-data, randomized-SVD, example, dataset, and public-API behavior are
  unchanged.
- Ordinary PLS remains available only where the maintained component-path comparison requires it.
- Future benchmark-like work requires a new decision based on a concrete package-validation need.
