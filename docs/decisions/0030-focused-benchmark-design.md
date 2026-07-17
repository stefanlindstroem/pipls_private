# Decision 0030: focused benchmark questions and minimal outputs

## Context

The first synthetic benchmark implementation combined several distinct questions—prediction,
rank selection, latent recovery, solver consistency, software versions, execution controls, and
runtime—into one manifest-driven runner and one wide CSV schema. The result was technically
machine-readable but difficult for a programming user to inspect and interpret.

## Decision

Replace the universal benchmark architecture with independent focused benchmarks.

Each benchmark must have one explicit user-facing question, one controlled setup, and one dedicated
CSV output containing only relevant columns. Initial benchmarks are:

1. fixed-structure recovery;
2. rank selection;
3. predictor-nuisance comparison with ordinary PLS;
4. solver consistency.

Software versions, execution controls, timings, and unrelated metrics are omitted unless they are
part of the benchmark's stated question. Runtime measurement, if later needed, receives its own
benchmark and hardware contract.

Remove the universal synthetic manifest, universal result schema, broad CI runner, and their tests.
Do not replace them in this patch. Implement the focused benchmarks separately, beginning with
fixed-structure recovery.

## Consequences

- Benchmark code becomes readable, question-specific repository code rather than generic
  orchestration infrastructure.
- Every output table has a small stable header that a human can understand without schema lookup.
- Tests may protect each benchmark's executable contract once implemented, but no broad shared row
  format is required.
- Decisions 0027–0029 remain historical records; their universal manifest/runner/schema consequences
  are superseded by this decision. Their general package-validation and human-readable CSV principles
  remain applicable where consistent with this focused design.
