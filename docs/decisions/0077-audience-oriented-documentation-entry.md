# Decision 0077: audience-oriented documentation entry

## Status

Accepted and implemented.

## Context

Decisions 0075 and 0076 established a short synthetic tutorial followed by a focused Pulp
real-data tutorial. The root README still contained extensive plotting code, validation commands,
documentation-build instructions, benchmark descriptions, distribution checks, and a repository
map. It therefore remained a second manual rather than a package landing page.

The served navigation also grouped user-facing datasets with maintainer-oriented benchmarks and
reproducibility material under one broad "Data and validation" section, while compatibility lived
inside the programming reference.

## Decision

Make the root README an orientation page for package users. It owns:

- a concise explanation of Pi-PLS and the two ranks;
- installation from a source checkout and optional user extras;
- one compact fixed-model workflow;
- one compact path-selection workflow;
- the ordered tutorial route;
- a small public-interface map and links to examples, datasets, API reference, and theory.

Move development environments, Make targets, documentation builds, distribution validation,
minimum-stack checks, generated-file policy, the repository map, and snapshot instructions to
`CONTRIBUTING.md`.

Organize the served site by audience:

- tutorials and examples for learning workflows;
- reference pages for programming contracts;
- project validation for reference-dataset provenance, benchmarks, reproducibility, and
  compatibility;
- scientific background for theory.

The documentation home page mirrors those routes without duplicating the detailed contracts.

## Consequences

The README becomes substantially shorter and no longer competes with the tutorials or contributor
documentation. Programming users can reach a fitted or selected model without reading repository
maintenance instructions. Maintainers retain one complete command and repository-ownership guide
in `CONTRIBUTING.md`.

Documentation Patch D4 is next: clean remaining references and enforce documentation ownership
without freezing exact prose.
