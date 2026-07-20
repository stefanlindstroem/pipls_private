# Decision 0049: public documentation source boundary

## Status

Accepted and implemented.

## Context

The repository already contained substantial Markdown guides, but some public pages linked to
`examples/`, `datasets/`, and `.llm/` outside the documentation source tree. The public theory page
was only a short summary and directed readers to maintainer contracts for the complete explanation.
Several pages also retained discussion of alternatives that were considered but never implemented.

A built documentation site needs one self-contained source boundary. User-facing documentation
should explain the method and the implemented package directly rather than reconstructing the
project's design conversation.

## Decision

- Treat `docs/` as the complete public documentation source.
- Do not require `.llm/`, repository scripts, example source files, or dataset README files to
  understand the public theory or API.
- Prohibit Markdown links under `docs/` whose target begins with `../`.
- Add public pages summarizing numbered examples and accepted design-decision navigation.
- Keep individual design records available as maintainer history, but keep them outside the primary
  user journey.
- Make `docs/theory.md` a self-contained account of the implemented Pi-PLS construction, rank
  interpretation, fitted quantities, numerical invariances, and validation consequences.
- Describe current implemented behavior. Do not document selection rules, parameters, or workflows
  that are not part of the package.

## Consequences

The public guides can be built from `docs/` without reaching into repository-maintenance files.
Examples and datasets remain visible through concise public summaries. Maintainer contracts may be
more detailed, but they are not user prerequisites. The next increment may add MkDocs and strict
link validation on top of this stable source boundary.
