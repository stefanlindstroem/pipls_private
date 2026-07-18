# Decision-record navigation

This file is a compact index for LLM-assisted work. The records under `docs/decisions/` contain the
accepted rationale and consequences. Read the full record whenever a change touches its subject.
This index is navigation, not a substitute for those records.

| Record | Subject | Implemented consequence |
|---|---|---|
| `0001-core-definition.md` | fixed Pi-PLS construction | SVD/least-squares core with explicit `(h, r_pi)` admissibility |
| `0002-preprocessing-semantics.md` | centering and scaling | preprocessing remains outside the fixed numerical core |
| `0003-predictor-rank-selection.md` | rank bound and conditional selection | ceiling rule, materialized splits, deterministic low-rank ties; sample-count convention refined by 0032 |
| `0004-response-standardized-mse.md` | selection loss | fold-local response scales and uniform response weighting |
| `0005-leave-one-out-protocol.md` | advanced validation | ordinary splitters, singleton-safe scoring, ordered OOF reporting |
| `0006-paper-versus-api-rank-rule.md` | publication versus general API | publication-specific rules remain external to estimator defaults |
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
| `0017-first-real-dataset.md` | former first transparent real-data integration | superseded by Decision 0033; the Linnerud integration is removed |
| `0018-repository-dataset-layout.md` | committed real-dataset file convention | every dataset uses comma-delimited `X.csv`, `Y.csv`, and documentary `metadata.yaml` |
| `0019-pulp-dataset-integration.md` | pulp dataset integration | pulp uses public supplementary provenance, CC BY 4.0 attribution, named column selection, and direct X/Y reading |
| `0020-public-dataset-provenance-boundary.md` | public versus internal dataset materials | public-only provenance; no private paths or preparation-only scripts; Corn preprocessing remains user-facing |
| `0021-durable-repository-tests.md` | repository test stability | test behavior and structural format without pinning living prose or documentary field values |
| `0022-sugarcane-dataset-integration.md` | sugarcane dataset integration | public LabSpec spectra, explicit row and wavelength selection, and compact regular-axis metadata |
| `0023-tobacco-dataset-integration.md` | tobacco dataset integration | public raw FT-NIR spectra, explicit ID alignment, metadata-column exclusion, and no spectral preprocessing |
| `0024-package-product-repository-boundary.md` | package versus publication ownership | `pipls` owns the software product; paper reproduction stays downstream; future block-aware API design is deferred |
| `0025-model-internal-standardization-boundary.md` | current versus future scaling | estimator centering/scaling is current and fold-local; only future block-aware variants are deferred |
| `0026-package-navigation-cleanup.md` | public repository navigation | remove paper placeholders and organize entry points around the installable software product |
| `0027-synthetic-benchmark-contract.md` | earlier broad synthetic benchmark contract | historical manifest/schema design superseded by Decision 0030 |
| `0028-synthetic-ci-benchmark-runner.md` | earlier broad CI runner | historical implementation removed by Decision 0030 |
| `0029-human-and-machine-readable-results.md` | benchmark result usability | CSV principle retained; universal wide schema superseded by Decision 0030 |
| `0030-focused-benchmark-design.md` | focused benchmark questions | one question, one script, and one minimal CSV output per benchmark |
| `0031-default-selection-support.md` | ordinary rank-selection defaults | five samples per retained predictor direction and five-fold CV by default |
| `0032-full-sample-rank-support.md` | rank-support sample-count convention | full supplied $n$ defines support; centered training folds impose feasibility caps |
| `0033-remove-linnerud-integration.md` | reference dataset scope | remove the Linnerud dataset, example, test, and active navigation |
| `0034-two-stage-component-path-workflow.md` | component-path presentation | one CSV row per component count, explicit predictor-rank policy, CSV-derived PDF, and separate fixed final fit |
| `0035-tobacco-randomized-auto-path.md` | former Tobacco solver demonstration | superseded by Decision 0036; randomized SVD remains covered by solver consistency |
| `0036-real-data-pls-path-comparison.md` | real-data example comparison | separate Pi-PLS and PLS CSV paths, overlaid PDF, and full-SVD Tobacco workflow |

## Accepted clarifications after earlier proposals

These points are fixed by implemented decisions and owner review even where the broad publication
plan contains an earlier or more general proposal:

- both adaptive public defaults use the name `"auto"`; exhaustive search is explicit `"optimal"`;
- both public selection interfaces default to `samples_per_predictor_rank=5` and `cv=5`;
- the samples-per-rank support term uses the total number of observations supplied to `fit()`,
  while centered training-fold dimensions remain hard candidate-feasibility caps;
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
- examples show their data-reading and matrix-construction code rather than relying on hidden
  utility functions;
- committed repository datasets use comma-delimited `X.csv`, `Y.csv`, and `metadata.yaml`, while
  external users remain free to use any data source or file organization;
- the current real-data integration suite is pulp, sugarcane, and tobacco; all are repository
  example data, not runtime loaders or publication-result claims;
- committed dataset assets use public or included provenance only; private archive references and
  preparation-only scripts stay outside the repository, while Corn raw-data preprocessing will be
  explicit and user-facing;
- repository tests validate executable behavior and durable file structure rather than pinning
  living roadmap prose or documentary metadata values;
- `pipls` is the long-lived package repository; paper figures, full experiment grids, paper-only
  OLS/CCA comparisons, cached results, and publication environments belong downstream;
- current estimator centering and optional scaling are integral to every fit, are learned inside
  each training fold during selection, and are refitted on all supplied training data;
- future block-aware scaling remains a valid direction, but only its API design and schedule are
  deferred until the owner starts a dedicated phase;
- the version-1 synthetic package benchmark uses ordinary PLS as its sole external comparator,
  separates oracle model validation from Pi-PLS selection validation, and freezes no predictive or
  performance claim before calibration;
- the universal benchmark manifest, wide result schema, and broad CI runner were removed; focused
  benchmarks now use one readable script and one minimal CSV output per user-facing question;
- real-data component paths are presented as CSV tables with fold SD, plots are derived from those
  tables, and examples fit a separate fixed model after an explicit user component choice;
- real-data examples compare separate Pi-PLS and standard PLS component-path CSVs in one PDF;
  Tobacco uses adaptive rank scanning with explicit full predictor SVD, while randomized SVD is
  covered by the solver-consistency benchmark;

When a new accepted architectural or public-API decision is added, create a numbered decision
record and add it to this index in the same patch.
