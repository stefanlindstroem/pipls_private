# Decision 0050: buildable documentation foundation

## Status

Accepted and implemented.

## Context

Decision 0049 made `docs/` the self-contained public documentation source. The repository still had
no documentation builder: `make docs` printed a placeholder, the `docs` optional dependency group
was empty, and no automated command checked navigation, internal links, or mathematical rendering.

The existing guides are Markdown-first and contain substantial inline and display mathematics. The
documentation foundation should therefore build those sources directly rather than converting them
to another authoring format.

## Decision

- Build the public site with MkDocs and the Material theme.
- Keep documentation dependencies in the dedicated `docs` optional dependency group.
- Run `mkdocs build --strict` through `make docs` so warnings fail the build.
- Define a user-oriented navigation rather than reproducing the repository directory tree.
- Keep accepted decision records built and linkable while excluding individual records from the
  primary navigation.
- Use PyMdown Extensions and MathJax to render the existing mathematical notation.
- Add generated API tooling only when the first generated reference pages are introduced, so the
  foundation has no unused documentation dependency.
- Write generated output to `site/`, exclude it from Git, and remove it through `make clean`.

## Consequences

A clean checkout can build a complete navigable site after installing `pipls[docs]`. Broken links,
missing navigation targets, invalid anchors, and other MkDocs warnings become build failures. The
source remains ordinary Markdown under `docs/`, and generated site files are not versioned.
