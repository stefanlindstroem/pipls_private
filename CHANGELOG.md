# Changelog

## Unreleased

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
