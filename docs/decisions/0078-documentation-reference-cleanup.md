# Decision 0078: documentation reference cleanup and ownership enforcement

## Status

Accepted and implemented.

## Context

Documentation Patches D1--D3 established a short synthetic entry tutorial, a focused Pulp
real-data tutorial, a concise README, contributor-owned maintenance instructions, and navigation
separated by audience. The remaining cleanup work was not another tutorial rewrite. Programming
users still lacked a compact result-object map and a task-oriented troubleshooting page, while
repository tests checked some exact explanatory phrases instead of the underlying documentation
contracts.

The strict MkDocs build catches broken links and anchors, but ordinary repository tests did not
validate those references generically. This allowed stale anchors to survive until a documentation
build was run.

## Decision

Complete the documentation sequence with four narrow changes.

1. Add one result-object map to the API overview. It identifies how each immutable public result is
   obtained and the programming question it answers.
2. Add a concise troubleshooting page under programming reference. It covers rank selection,
   candidate feasibility, `refit=False`, custom scoring, grouped splitters, failed fits,
   `StatisticalSupportWarning`, `copy=False`, and the optional plotting dependency.
3. Add a generic served-Markdown link and anchor test. It resolves local files, ordinary heading
   anchors, explicit anchors, and generated mkdocstrings object anchors without copying prose into
   tests.
4. Remove tests that pin tutorial sentences or navigation display wording when snippets, assets,
   public objects, links, and workflow order provide a more durable contract.

Served guides must not expose internal phase or patch labels. Maintainer history remains in
`docs/decisions/` and `.llm/`, outside the served user route.

## Consequences

Programming users can identify the appropriate result object and recover from common public-API
problems without reading private implementation material. Broken local documentation links and
anchors fail in the ordinary test suite as well as in strict MkDocs builds. Tutorial wording may
evolve while executable snippets, generated assets, result ownership, and workflow order remain
protected.

Documentation Patches D1--D4 are complete. First-release preparation is the next repository
increment.
