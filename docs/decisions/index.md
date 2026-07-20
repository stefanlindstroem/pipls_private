# Design decisions

The decision records document accepted repository contracts for maintainers. They are historical
engineering records, not prerequisites for using Pi-PLS. The user guides and generated API
reference describe the current public behavior directly.

## Numerical method and estimator API

- [0001: core definition](0001-core-definition.md)
- [0002: preprocessing semantics](0002-preprocessing-semantics.md)
- [0003: predictor-rank selection](0003-predictor-rank-selection.md)
- [0004: response-standardized MSE](0004-response-standardized-mse.md)
- [0005: leave-one-out protocol](0005-leave-one-out-protocol.md)
- [0006: publication and API rank rules](0006-paper-versus-api-rank-rule.md)
- [0007: predictor-rank search policies](0007-predictor-rank-search-policies.md)
- [0008: predictor SVD policy](0008-predictor-svd-policy.md)
- [0009: public parameter validation](0009-public-parameter-validation.md)
- [0010: path-analysis API](0010-path-analysis-api.md)
- [0011: shared selection engine](0011-shared-selection-engine.md)
- [0012: scikit-learn API alignment](0012-sklearn-api-alignment.md)
- [0013: scikit-learn cleanup boundary](0013-sklearn-cleanup-boundary.md)
- [0025: model-internal standardization](0025-model-internal-standardization-boundary.md)
- [0031: default selection support](0031-default-selection-support.md)
- [0032: full-sample rank support](0032-full-sample-rank-support.md)
- [0039: fixed estimator and path search](0039-fixed-estimator-path-search-boundary.md)
- [0040: scikit-learn API polish](0040-sklearn-api-polish.md)

## Validation, data, and benchmarks

- [0014: validation metadata scope](0014-validation-metadata-scope.md)
- [0015: dataset and synthetic API](0015-dataset-and-synthetic-api.md)
- [0016: transparent data ingestion](0016-transparent-data-ingestion.md)
- [0017: first real dataset](0017-first-real-dataset.md)
- [0018: repository dataset layout](0018-repository-dataset-layout.md)
- [0019: Pulp integration](0019-pulp-dataset-integration.md)
- [0020: public dataset provenance](0020-public-dataset-provenance-boundary.md)
- [0022: Sugarcane integration](0022-sugarcane-dataset-integration.md)
- [0023: Tobacco integration](0023-tobacco-dataset-integration.md)
- [0027: synthetic benchmark contract](0027-synthetic-benchmark-contract.md)
- [0028: synthetic CI benchmark runner](0028-synthetic-ci-benchmark-runner.md)
- [0029: human- and machine-readable results](0029-human-and-machine-readable-results.md)
- [0030: focused benchmark design](0030-focused-benchmark-design.md)
- [0033: remove Linnerud integration](0033-remove-linnerud-integration.md)
- [0035: former Tobacco randomized path](0035-tobacco-randomized-auto-path.md)
- [0041: legacy dataset licensing](0041-legacy-dataset-licensing-roadmap.md)

## Examples and analysis workflows

- [0034: two-stage component-path workflow](0034-two-stage-component-path-workflow.md)
- [0036: real-data PLS path comparison](0036-real-data-pls-path-comparison.md)
- [0037: user-run real-data analyses](0037-user-run-real-data-analyses.md)
- [0038: examples target](0038-single-examples-target.md)
- [0042: model inspection and post-analysis](0042-model-inspection-and-post-analysis.md)
- [0043: PLS biplot and analysis surface](0043-pls-biplot-and-analysis-surface.md)
- [0044: minimal onboarding and example support](0044-minimal-onboarding-and-example-support.md)
- [0045: PLS-family analysis boundary](0045-pls-family-analysis-boundary.md)
- [0046: concise numbered examples](0046-concise-numbered-examples.md)
- [0047: separate PLS path comparison](0047-separate-pls-path-comparison-example.md)
- [0048: self-contained user examples](0048-self-contained-user-examples.md)

## Repository and documentation policy

- [0021: durable repository tests](0021-durable-repository-tests.md)
- [0024: package product boundary](0024-package-product-repository-boundary.md)
- [0026: package navigation cleanup](0026-package-navigation-cleanup.md)
- [0049: public documentation source boundary](0049-public-documentation-source-boundary.md)
