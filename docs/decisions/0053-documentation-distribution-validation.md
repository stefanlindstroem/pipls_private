# Decision 0053: documentation distribution validation

## Status

Accepted and implemented.

## Context

The strict documentation site builds from a repository checkout, and the generated reference covers
the complete supported public API. A source distribution must also contain the MkDocs configuration,
Makefile, documentation sources, JavaScript assets, and package source needed to reproduce that
site. Building only from a checkout does not verify this distribution boundary.

Generated `site/` output is disposable. It must remain outside Git, source distributions, and
repository snapshots.

## Decision

- Include `Makefile`, `mkdocs.yml`, the documentation-validation helper, and all documentation
  sources and assets in the source distribution.
- Add `make docs-dist`, which builds an sdist, unpacks it, creates a clean virtual environment,
  installs that unpacked source with its `docs` extra, and runs the strict documentation build.
- Validate representative generated guide and API pages after the sdist build.
- Run both `make docs` and `make docs-dist` in a dedicated CI job.
- Keep the ordinary unit-test suite independent of documentation downloads and clean-environment
  installation work.
- Exclude generated `site/` output from Git, source distributions, and repository snapshots.

## Consequences

Documentation can be reproduced from the artifact delivered to source-distribution users, not only
from the development checkout. Missing build configuration or documentation assets now fail CI.
The stronger distribution check is explicit because it creates a temporary environment and may
install dependencies; `make check` remains the fast code-quality boundary.
