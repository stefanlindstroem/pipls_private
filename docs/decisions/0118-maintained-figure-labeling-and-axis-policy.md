# Decision 0118: maintained figure labeling and axis policy

## Status

Accepted and implemented.

## Context

The maintained example and documentation figures had accumulated inconsistent method notation,
factor-element symbols, subplot titles, and vertical scales. Several tiled figures repeated their
axis labels as subplot titles. Some response-factor labels used lower-case $q$ even though they
refer to elements of the matrix $Q$. Component paths and predictor-rank profiles did not all use the
same zero-based CV-MSE scale. The dense Tobacco figures also needed explicit categorical-label
rotation.

## Decision

Apply one rendering policy to every maintained numbered-example PDF and generated tutorial SVG:

- use `$\Pi$`-PLS whenever the method name appears in rendered figure text;
- use upper-case $P$ and $Q$ for factor-matrix elements and lower-case $d$ for diagonal elements of
  $D$;
- omit subplot titles from tiled Pi-PLS factor, latent-structure, and prediction-diagnostic figures;
- keep prediction-diagnostic figure-level titles on one rendered line;
- give every maintained component path and predictor-rank profile a lower y-limit of zero and an
  upper y-limit of at least one;
- rotate the Tobacco factor-component, prediction-response, and latent-response labels where
  requested;
- omit single-plot titles that merely repeat the quantity already stated by the y-axis label.

This policy belongs to the caller-owned example and documentation rendering layer. It does not add a
package plotting API or change numerical results.

## Consequences

- Regenerating `make examples` visibly updates every maintained example figure family affected by
  the policy.
- Regenerating `make docs-figures` visibly updates all synthetic tutorial figures and all Pulp
  tutorial figures.
- Figure labels follow the notation used by the implemented factorization.
- Source-level rendering-policy tests protect the complete maintained renderer set.
