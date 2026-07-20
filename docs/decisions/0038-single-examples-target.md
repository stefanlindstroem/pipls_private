# Decision 0038: one explicit target runs every example

## Status

Accepted and implemented.

## Context

Decision 0037 removed complete real-data analyses from `make check` because those executions are
slow, hardware-dependent, and duplicate neither unit tests nor focused synthetic benchmarks. The
repository still needs one clear command for validating the application-facing examples and
regenerating their CSV and PDF artifacts.

Splitting the examples into fast and slow make targets would require users to know which analyses
were omitted from the shorter target. That is unnecessary ambiguity for an explicit opt-in command.

## Decision

The Makefile provides one phony target:

```bash
make examples
```

It runs every numbered script under `examples/` in order:

1. minimal literal-matrix fit and plot;
2. explained deterministic synthetic train/test data;
3. dedicated Pulp, Sugarcane, and Tobacco Pi-PLS-versus-PLS path comparisons;
4. Pulp Pi-PLS path and fixed-model analysis;
5. Sugarcane Pi-PLS path and fixed-model analysis;
6. Tobacco Pi-PLS path and fixed-model analysis.

The target sets `PYTHONPATH=src` and `MPLBACKEND=Agg` for each command so a source checkout and a
headless shell can execute the examples consistently. It also limits common BLAS/OpenMP backends to
one thread per process, avoiding nested numerical-thread oversubscription inside the examples.

`make examples` is not a dependency of `make check`. It is an explicit application-validation
command and may take substantially longer because it includes the full Tobacco analysis.

## Consequences

- There is no partial or additional examples target.
- A successful `make examples` means every shipped numbered example completed.
- Changes to numbered examples, example CSV/PDF generation, or the application-facing workflow
  should validate with `make examples`.
- Fast internal validation remains `make check`.
- Generated files remain under `examples/results/` and are ignored by Git.

## Later refinement

Decision 0048 removes the context-free advanced-cross-validation script while retaining the single
`make examples` target. Advanced splitters remain documented and supported.
