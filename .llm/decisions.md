# Decision-record navigation

This file is a compact index for LLM-assisted work. The records under `docs/decisions/` contain the
accepted rationale and consequences. Read the full record whenever a change touches its subject.
This index is navigation, not a substitute for those records.

| Record | Subject | Implemented consequence |
|---|---|---|
| `0001-core-definition.md` | fixed Pi-PLS construction | SVD/least-squares core with explicit `(h, r_pi)` admissibility |
| `0002-preprocessing-semantics.md` | centering and scaling | preprocessing remains outside the fixed numerical core |
| `0003-predictor-rank-selection.md` | rank bound and conditional selection | fold-safe ceiling rule, materialized splits, deterministic low-rank ties |
| `0004-response-standardized-mse.md` | selection loss | fold-local response scales and uniform response weighting |
| `0005-leave-one-out-protocol.md` | advanced validation | ordinary splitters, singleton-safe scoring, ordered OOF reporting |
| `0006-paper-versus-api-rank-rule.md` | reproduction versus general API | paper rules remain explicit reproduction inputs, not estimator defaults |
| `0007-predictor-rank-search-policies.md` | exhaustive versus adaptive search | `"optimal"` is exhaustive; `"auto"` is deterministic approximate search |
| `0008-predictor-svd-policy.md` | scalable predictor decomposition | independent `full`, `randomized`, and `auto` solver policy |
| `0009-public-parameter-validation.md` | exposed controls | early validation and low-statistical-support warning |
| `0010-path-analysis-api.md` | triangular path search | pipeline-aware `PiPLSPathCV` with aligned search vocabulary |
| `0011-shared-selection-engine.md` | code ownership | both public interfaces use the same private fold/search machinery |
| `0012-sklearn-api-alignment.md` | estimator and PLS compatibility | standard fitted surface plus structured Pi-PLS decomposition output |
| `0013-sklearn-cleanup-boundary.md` | final pre-D2 scope | direct estimator or terminal-pipeline support and conditional delegation |
| `0014-validation-metadata-scope.md` | groups and weighting boundary | groups-only splitter metadata; no weighted fitting or general routing |
| `0015-dataset-and-synthetic-api.md` | dataset and synthetic boundary | optional immutable datasets plus local seeded latent-structure generation |
| `0016-transparent-data-ingestion.md` | real-data and example boundary | users and examples read `X` and `Y` explicitly; no required registry or generic loader |
| `0017-first-real-dataset.md` | first transparent real-data integration | Linnerud files are read explicitly; provenance remains repository-only |
| `0018-repository-dataset-layout.md` | committed real-dataset file convention | every dataset uses comma-delimited `X.csv`, `Y.csv`, and documentary `metadata.yaml` |

## Accepted clarifications after the original publication plan

These points are fixed by implemented decisions and owner review even where the broad publication
plan contains an earlier or more general proposal:

- both adaptive public defaults use the name `"auto"`; exhaustive search is explicit `"optimal"`;
- randomized SVD is controlled independently and follows the same policy inside regression and
  path candidate fits;
- `PiPLSRegression` is not implemented as a wrapper around `PiPLSPathCV`; both use private shared
  machinery;
- `PiPLSPathCV` supports a direct estimator or a scikit-learn `Pipeline` ending in
  `PiPLSRegression`, not arbitrary nested meta-estimators;
- D2 supports explicit group metadata for splitters, but weighted fitting and general-purpose
  sample metadata routing are not project goals;
- repeated and partial-coverage OOF predictions are a documented Pi-PLS extension rather than a
  claim of exact `cross_val_predict` equivalence;
- selection-conditioned validation reports are descriptive diagnostics, not unbiased nested-CV or
  external-test estimates;
- real-data users supply `X` and `Y` directly; metadata files, registry lookup, generic loaders,
  and `PiPLSDataset` are not prerequisites for fitting;
- examples and reproduction scripts show their data-reading and matrix-construction code rather
  than relying on hidden utility functions;
- committed repository datasets use comma-delimited `X.csv`, `Y.csv`, and `metadata.yaml`, while
  external users remain free to use any data source or file organization;
- the first real-data integration is Linnerud; it is repository example data, not a new runtime loader or a manuscript-result claim;

When a new accepted architectural or public-API decision is added, create a numbered decision
record and add it to this index in the same patch.
