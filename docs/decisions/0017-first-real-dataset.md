# Decision 0017: first transparent real-dataset integration

## Status

Superseded by Decision 0033. The integration described below has been removed.

## Context

The first real-data patch must exercise the repository's transparent input contract without
introducing a public loader, metadata requirement, or hidden example utility. It should be small
enough to audit completely and have resolved source, citation, license, redistribution, row-order,
and missing-value properties.

## Decision

- The Linnerud physical-exercise dataset is the first repository real-data integration.
- The values originate from scikit-learn 1.8.0 and are redistributed under its BSD 3-Clause
  license.
- Predictor and response tables remain separate, preserve their documented common row order, and
  are normalized to the repository-wide `X.csv` and `Y.csv` names and comma-delimited format.
- `metadata.yaml` records source, license, dimensions, variables, alignment, preparation, and
  integrity without becoming a runtime requirement.
- The example reads both tables directly with pandas, validates columns, numeric dtypes, row counts,
  and missingness, then fits `PiPLSRegression`.
- No package loader, registry, metadata-driven runtime path, automatic download, or hidden
  example helper is added.
- Integrity hashes and provenance documentation are repository assets only; they are not runtime
  requirements.
- Repository datasets and examples are included in the source distribution but not in the wheel.

## Consequences

- A fresh user-facing example demonstrates the exact `read X`, `read Y`, `fit(X, Y)` workflow.
- The first E3 migration validates the repository contract without entangling the public estimator
  API with data management.
- Manuscript datasets remain separate later migrations with their own scientific and licensing
  decisions.
