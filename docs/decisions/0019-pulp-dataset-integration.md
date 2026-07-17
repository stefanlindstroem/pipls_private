# Decision 0019: pulp dataset integration

## Status

Accepted during Phase E3.

## Context

The supplied `PiPLSR_v0.1` archive contains several manuscript datasets. The next integration must
be reviewable, redistributable, and scientifically explicit. The pulp source is a compact CSV with
no missing values, and its variable structure is documented by an open-access 2025 publication and
supplementary material. The archived analysis code already states the exact Pi-PLS column split.

## Decision

- Integrate pulp as the second repository real dataset.
- Use source columns `Shives` through `F (llw)` as the 14 predictors.
- Use `CSF` through `s` as the eight responses.
- Exclude refiner controls, specific refining energy, and the final `k` column from the model
  matrices, matching the supplied analysis code.
- Preserve all 46 rows and all selected numeric values without filtering, scaling, or imputation.
- Record the supplied source checksum and use a deterministic dataset-specific preparation script.
- Attribute the associated article and supplementary material under CC BY 4.0.
- Keep the public workflow transparent: the example reads `X.csv` and `Y.csv` directly and does not
  parse metadata or call a package loader.

## Consequences

- The repository gains a manuscript-related multivariate dataset without broadening the runtime
  API.
- The analytical column selection is reviewable rather than hidden in a loader.
- Future reproduction work can refer to the exact source checksum and preparation script.
- This integration does not claim reproduction of published figures or benchmark performance.
