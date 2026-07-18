# Decision 0037: real-data analyses are user-run examples

## Status

Accepted and implemented.

## Context

Examples 10, 11, and 12 already run the complete Pulp, Sugarcane, and Tobacco component-path
analyses. They write canonical Pi-PLS and standard PLS CSV files, generate a comparison PDF, and fit
a separately chosen fixed Pi-PLS model.

Separate real-data smoke benchmark scripts repeated the Pi-PLS portion of those analyses. Their
benchmark tests repeated the same path computations again, while dataset-specific tests executed the
complete examples, including CSV/PDF generation and final fitting. Tobacco made this duplication
particularly expensive and hardware-dependent.

## Decision

The Pulp, Sugarcane, and Tobacco smoke benchmark scripts and their tests are removed. Tests that
execute the complete real-data examples are also removed.

The examples remain executable, documented user analyses. Users run them explicitly when they need
the dataset-specific CSV and PDF artifacts. They are not part of `make check`.

Default tests retain durable contracts through:

- generic repository-dataset layout and numeric-readability tests;
- component-path API tests on small synthetic data;
- standard PLS path-helper equivalence tests;
- CSV-to-PDF artifact tests;
- static checks that examples retain the two-stage workflow and required solver/search choices;
- the four focused synthetic benchmark suites.

A repository-boundary test protects the absence of the retired real-data benchmark copies and full
example-execution tests.

## Consequences

- `make check` no longer performs complete real-data analyses.
- Running an example is an explicit analysis action rather than a hidden validation cost.
- Dataset assets and public examples remain installed and documented.
- The synthetic benchmark layer remains focused on four distinct numerical questions.
- Future real-data benchmarks require a package-validation question that is not already answered by
  a public example or a small durable contract test.
