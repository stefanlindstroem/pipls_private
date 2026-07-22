# Decision 0076: focused Pulp tutorial

## Status

Accepted and implemented.

## Context

Decision 0075 added a short synthetic tutorial that now owns the minimum component-path,
conditional predictor-rank, fixed-fit, and external-test prediction workflow. The Pulp tutorial no
longer needs to teach that contract from the beginning.

The Pulp page still repeated introductory selection material, carried common estimator variations
already owned by reference pages, and displayed thirteen generated figures. It therefore continued
to function partly as a plotting catalogue even though `docs/model_inspection.md` already owns the
complete interpretation reference.

## Decision

Reposition `docs/tutorials/pulp.md` as the second-stage real-data tutorial. It assumes the synthetic
tutorial and concentrates on:

- explicit Pulp data loading and provenance;
- the real-data component choice and upper-boundary predictor-rank qualification;
- fitting the selected fixed model;
- selection-conditioned OOF predictions and their interpretation boundary;
- immutable inspection results;
- representative standard PLS-family and Pi-PLS-specific plots.

Retain six generated tutorial figures:

```text
component_path.svg
predictor_rank_profile.svg
biplot.svg
predictor_directions.svg
observed_vs_predicted.svg
standardized_rmse.svg
```

The complete model-inspection catalogue remains in `docs/model_inspection.md`, and the numbered Pulp
example continues to demonstrate the broader plotting API and write its existing six PDF reports.
Remove the Pulp tutorial's common-variations section; advanced search, validation, preprocessing,
refit, and scorer behavior remain in the generated API and specialized reference pages.

Add an explicit stopping point after the fixed fit, OOF predictions, and immutable inspection
results. Readers interested only in the programming workflow should not need to traverse the plot
interpretation section.

## Consequences

The Pulp tutorial becomes shorter and has one clear purpose: complete a scientifically qualified
real-data analysis after the controlled synthetic introduction. Generated documentation work is
reduced from thirteen to six Pulp SVG figures without removing any public plotting function or
numbered-example output.

Documentation Patch D3 is next: reduce the README and finish the audience-oriented navigation.
