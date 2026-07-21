# Decision 0057: public documentation entry and maintainer-record exclusion

## Status

Accepted and implemented.

## Context

The served documentation exposed the complete maintainer decision history and introduced examples
with terms such as component path and $P D Q^{\mathsf T}$ before explaining the purpose of Pi-PLS.
The compatibility page also repeated implementation and CI rationale that users do not need when
checking whether their environment is supported.

The public site must introduce the method before specialized terminology, while the repository must
retain its engineering history for maintainers.

## Decision

- Exclude `docs/decisions/` from the MkDocs build and search index. Keep the records in the
  repository and source distribution for maintainer use.
- Begin the public documentation with a short account of Pi-PLS as multivariate regression through
  paired predictor-response latent variables.
- Define component count, predictor rank, and component path before using those terms in the
  examples or selection guides.
- Present the routine selection workflow as a cross-validated scan of CV-MSE against component
  count, followed by a parsimonious choice such as an elbow or plateau and a separate fixed fit.
- Reserve matrix-factor notation for the theory and inspection pages, with links from introductory
  material where it is relevant.
- Keep the compatibility page to the supported interpreter range, dependency bounds, minimum tested
  stack, and a brief statement of the checks used to maintain those bounds.

## Consequences

The served site is user-facing rather than an engineering archive. A reader can understand the
purpose of the examples before encountering implementation-specific names, while the detailed
mathematical construction remains available through explicit theory links. Maintainer decisions
remain versioned and distributed but are not rendered or searchable in the public site.
