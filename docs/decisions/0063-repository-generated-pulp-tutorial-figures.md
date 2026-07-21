# Decision 0063: repository-generated Pulp tutorial figures

## Status

Accepted and implemented.

## Context

The planned Pulp tutorial needs figures that agree with the executable analysis and remain available
when the documentation is built from a source distribution. Manually maintained screenshots would
drift from the canonical workflow and package plotting behavior. Reusing the numbered example's PDF
report would also couple the tutorial to a multipage artifact rather than to the individual charts
explained by the documentation.

## Decision

`tools/render_pulp_tutorial.py` calls the canonical workflow from
`examples/_support/pulp_workflow.py` and generates one SVG per tutorial chart under
`docs/assets/generated/pulp/`. The renderer owns figure dimensions, titles, legends, selected
components and responses, saving, and closing; package plotting functions remain one-axis chart
primitives.

The generated set contains the component path, scores, biplot, X and Y loadings, the four Pi-PLS
factor views, regression coefficients, observed-versus-predicted values, residuals, and standardized
RMSE. Response-heavy charts use the declared `CSF`, `Density`, and `TI` subset, while standardized
RMSE displays all responses. The component-path chart marks the stated three-component choice and
the predictor rank selected for that row.

`manifest.json` records the dataset hashes, selected rank pair, displayed components and responses,
prediction provenance, generated filenames, and SVG hashes. It contains no timestamp. SVG metadata
omits the creation date and uses a fixed Matplotlib SVG hash salt.

`make docs-figures` regenerates the assets. `make docs` and `make docs-serve` depend on that target.
The generated directory is ignored by Git and removed by `make clean`. The source distribution ships
the renderer, canonical workflow, Pulp data, and plotting source, then regenerates and validates the
assets during `make docs-dist`.

## Consequences

The tutorial can include repository-derived figures without committing generated graphics. The
numerical workflow remains single-sourced, and figure composition remains outside the installed
package. Documentation builds now require pandas and Matplotlib in addition to the existing MkDocs
toolchain. The next documentation increment may write the tutorial against these stable filenames
and manifest fields without duplicating analysis code.
