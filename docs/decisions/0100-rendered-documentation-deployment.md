# Decision 0100: rendered documentation deployment

## Context

The root README routed users primarily to Markdown source files. Those files are useful in a source
checkout, but GitHub does not expand mkdocstrings directives or provide the generated tutorial
figures that belong to the built MkDocs site. The repository already validated strict documentation
from the checkout and source distribution, but it had no maintained deployment.

The snapshot does not contain a stable public repository URL that can be hard-coded safely. GitHub
Actions does, however, provide the repository owner and name when the site is built.

## Decision

1. Build the strict documentation from every push and pull request in a dedicated documentation
   workflow, including the clean source-distribution documentation check.
2. On pushes to `master`, derive the default GitHub Pages site and repository URLs from the Actions
   repository context, rebuild with those canonical values, upload `site/`, and deploy it through
   the official GitHub Pages actions.
3. Grant write and identity-token permissions only to the deployment job and serialize Pages
   deployments through one non-cancelling concurrency group.
4. Present the current `github-pages` deployment as the primary documentation route in the README,
   while retaining source links for offline checkouts. A future owner-confirmed permanent URL may
   replace the repository-relative deployment link without changing the deployment contract.
5. Keep generated `site/`, tutorial assets, and the temporary Pages configuration outside Git.

## Consequences

Pull requests validate the same strict user documentation that is deployed. A successful push to
`master` produces a rendered site with expanded API reference, generated figures, repository and
edit links, and canonical Pages URLs without embedding an unconfirmed owner name in the source.
Package publication and versioning remain outside this policy.
