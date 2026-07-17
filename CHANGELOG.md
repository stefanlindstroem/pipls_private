# Changelog

## Unreleased

- Correct the Pulp example and smoke check to make all 14 predictor ranks available explicitly;
  the conservative default samples-per-rank rule restricted the previous small-data search to rank
  4 and produced a misleading validation result.
- Add the transparent Pulp path-selection smoke check using direct pandas `X.csv`/`Y.csv` reading,
  the ordinary public `PiPLSPathCV` workflow, complete ordered five-fold OOF reporting, and a
  one-row CSV whose validation diagnostics are explicitly labeled selection-conditioned.
- Add the full-versus-randomized solver-consistency benchmark with paired fixed ranks, three high-dimensional matrix geometries, independent-test prediction and coefficient relative differences, a minimal four-column CSV, and focused contract tests.
- Add the paired predictor-nuisance benchmark comparing fixed Pi-PLS and ordinary PLS on identical deterministic synthetic train/test problems, with controlled nuisance strengths, training-fitted model standardization, independent-test MSE, a minimal five-column CSV, and focused contract tests.
- Add the adaptive rank-selection benchmark with deterministic public synthetic train/test data, generator-declared reference ranks, fold-local model standardization, full-training refit, independent-test MSE, a minimal seven-column CSV output, and focused repeatability tests.
- Add the fixed-structure recovery benchmark with deterministic public synthetic train/test data, fixed oracle Pi-PLS ranks, test MSE, coordinate-correct subspace captures, a minimal six-column CSV output, and focused repeatability tests.
- Replace the over-general synthetic benchmark manifest, universal schema, and broad CI runner with a focused benchmark plan: one user question and one minimal CSV output per benchmark.

- Remove paper-reproduction placeholders and reorganize public navigation around the installable package, user documentation, concise examples, transparent datasets, and lightweight validation responsibilities.

- Add the CC BY 4.0 tobacco FT-NIR dataset as the fourth transparent real-data integration, with public sample-ID alignment, 1,557 raw spectral predictors, 13 chemical responses, no spectral preprocessing, and direct pandas I/O.

- Add the CC BY 4.0 sugarcane LabSpec dataset as the third transparent real-data integration, with explicit sample alignment, missing-response exclusion, wavelength selection, compact spectral-axis metadata, and direct pandas I/O.

- Replace brittle repository-document and dataset-metadata value assertions with structural file, format, decision-index, and executable-example tests; document the durable testing policy.

- Restrict committed dataset provenance to public or included sources, remove private pulp archive references and its preparation-only converter, and document Corn as the explicit user-facing raw-data preprocessing exception.

- Add the CC BY 4.0 pulp dataset as the second transparent real-data integration, with explicit column selection, deterministic preparation, direct pandas I/O, provenance, integrity tests, and source-distribution packaging.

- Standardize committed real datasets on comma-delimited `X.csv`, `Y.csv`, and documentary `metadata.yaml`; normalize the Linnerud integration and examples to that convention.

- Add the first transparent real-data integration using the BSD-licensed Linnerud multi-output regression tables, explicit pandas I/O, provenance, checksums, tests, and source-distribution packaging.
- Replace the planned public dataset registry/loader with a transparent real-data contract: users and examples read `X` and `Y` explicitly, while repository preparation remains dataset-specific.
- Modernize package license metadata to the PEP 639 SPDX form and require a compatible setuptools build backend.
- Add the validated immutable dataset container and deterministic shared/predictor-specific/response-specific synthetic generators with leakage-free train/test construction.
- Establish repository skeleton and LLM-assisted snapshot/patch workflow.
- Add the fixed-parameter Pi-PLS numerical core.
- Add the public fixed-rank `PiPLSRegression` estimator.
- Add the ceiling-based predictor-rank bound and `predictor_rank="max"` mode.
- Add private cross-validation selection primitives for reusable splits, response-standardized loss, and deterministic rank tie-breaking.
- Add automatic predictor-rank selection with fold-local preprocessing, diagnostics, public scoring utilities, and full-data refitting.
- Implement exhaustive `"optimal"` rank search and deterministic adaptive coarse-to-fine `"auto"` search, with cached evaluations and search diagnostics.
- Add full, randomized, and conservative automatic predictor-SVD policies with reproducible seeds and fitted solver diagnostics.
- Harden public parameter validation and warn when rule-based predictor-rank bounds use fewer than five samples per retained direction.
- Add pipeline-aware `PiPLSPathCV` with exhaustive and adaptive triangular path search, fold-safe bounds, diagnostics, and full-data refitting.
- Align the public estimators with scikit-learn and `PLSRegression` conventions, including estimator-aware validation, feature names, pandas output, weights/loadings, standard CV results, `PiPLSDecomposition`, `best_pipls_`, and container-preserving path folds.
- Make pandas a required development dependency so pandas API-alignment tests run rather than skip.
- Add D2 advanced split protocols, group metadata routing, ordered OOF predictions, pooled OOF diagnostics, singleton-safe LOO handling, and explicit selection-conditioned validation reports.
- Complete the pre-D2 scikit-learn cleanup with PLS-style inverse transforms, canonical read-only decomposition aliases, standard CV sentinels and timing diagnostics, conditional path delegation, an explicit direct-or-final-pipeline estimator scope, and minimum-version CI.
