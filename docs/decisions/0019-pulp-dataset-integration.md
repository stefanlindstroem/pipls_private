# Decision 0019: pulp dataset integration

## Status

Accepted during Phase E3; public provenance clarified by Decision 0020.

## Context

The pulp dataset is a compact multivariate regression example whose variables are described by a
public 2025 article and associated supplementary material. The integration must be reviewable,
redistributable, scientifically explicit, and understandable from public repository materials
alone.

## Decision

- Integrate pulp as the second repository real dataset.
- Use the named variables `Shives` through `F (llw)` as the 14 predictors.
- Use `CSF` through `s` as the eight responses.
- Exclude refiner controls, specific refining energy, and the final `k` variable from the model
  matrices.
- Preserve all 46 rows and all selected numeric values without filtering, scaling, or imputation.
- Cite the associated article and supplementary material by DOI under CC BY 4.0.
- Describe the named selections directly in public metadata and licensing material.
- Do not publish private archive names, inaccessible source paths, unshared source checksums, or a
  preparation-only script.
- Keep the public workflow transparent: the example reads `X.csv` and `Y.csv` directly and does not
  parse metadata or call a package loader.

## Consequences

- The repository gains a manuscript-related multivariate dataset without broadening the runtime
  API.
- The analytical column selection is reviewable from public repository assets.
- Programming users are not directed toward unavailable files or internal project tooling.
- This integration does not claim reproduction of published figures or benchmark performance.
