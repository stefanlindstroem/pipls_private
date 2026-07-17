# Decision 0026: Package repository navigation

Status: accepted and implemented.

## Context

The repository had already adopted a long-lived package-product boundary, but still contained
`paper/` and `scripts/reproduce_paper/` placeholders and public text that described future
manuscript migration and paper reproduction. Those entry points implied that the package repository
would eventually become the reproduction environment for one publication.

The intended audience is broader: programming users installing and applying `pipls`, maintainers
extending its public API, and future publications depending on tagged package releases.

## Decision

- Remove the paper-reproduction placeholder directories.
- Organize the public README and documentation index around installation, model fitting, parameter
  selection, validation, examples, datasets, theory, reproducibility, and repository maintenance.
- Treat concise examples and lightweight benchmarks as package-validation and user-learning assets,
  not as manuscript figure or result pipelines.
- Preserve historical scientific context and accepted decision records where they explain current
  behavior.
- Keep complete simulations, paper-only comparators, cached publication results, tables, and figure
  generation in downstream repositories that pin released `pipls` versions.
- Make the synthetic benchmark contract the next package increment.

## Consequences

The repository has a coherent package-facing entry surface without changing runtime behavior. Tests
may protect the durable absence of retired paper scaffolding, but should not pin living navigation
prose. Future block-aware standardization remains deferred and is unaffected by this cleanup.
