# Changelog

## Unreleased

- Replace ordinary-PLS-specific shared inspection and plotting names with estimator-neutral PLS-family names, and accept compatible fitted `PiPLSRegression` and scikit-learn `PLSRegression` models through a structural public-attribute contract.
- Accept the corrected fitted-model analysis boundary: retain ordinary PLS in the shared CV-MSE
  component-path comparison, keep $P$, $D$, and $Q$ inspection explicitly Pi-PLS-specific, and
  require estimator-neutral shared PLS-family tools that numbered examples apply only to the
  selected Pi-PLS model. Runtime migration follows in separate patches.

- Add a literal-matrix `01_minimal_fit_and_plot.py` quickstart, organize user documentation around
  direct fit, selection, inspection, and complete reports, and move advanced real-data helper modules
  under `examples/_support/` without adding another Make target or changing estimator behavior.

- Complete the fitted-model analysis series with balanced reconstruction-preserving ordinary PLS biplot coordinates, a Pulp-only score-loading biplot reconstructed from canonical score/loading tables, and a final cross-dataset API and artifact review.

- Complete the Tobacco post-analysis workflow with decreasing-wavenumber spectral displays,
  selection-conditioned Pi-PLS and ordinary PLS OOF predictions, deterministic source-order
  response pagination, an optional eighth canonical CSV for raw PLS score-distance and
  X-reconstruction-residual diagnostics, and a CSV-derived multipage report without theoretical
  outlier limits.
- Remove the superseded standalone `09_model_inspection.py` workflow and its dedicated structural
  test. Pulp and Sugarcane now provide the maintained demonstrations of CSV-header label
  acquisition, fitted-model interpretation, OOF diagnostics, canonical tables, and reports;
  `make clean` removes any locally retained `examples/results/model_inspection/` directory.
- Align documentation and the guide layer with the restored ceiling-based path bound and the
  direct fixed-fit support warning at fewer than three observations per retained predictor-rank
  direction.
- Add the complete Sugarcane spectral post-analysis workflow: explicit wavelength coordinates
  from `X.csv` headers, fixed Pi-PLS and ordinary PLS OOF predictions, seven canonical long-form
  CSV tables, and a seven-page report with spectral directions, loadings, and coefficients rendered
  as physical-axis lines.
- Add the complete Pulp post-analysis workflow: fixed Pi-PLS and ordinary PLS OOF predictions
  on the established five folds, explicit selection-conditioned provenance, seven canonical
  long-form CSV tables, and a seven-page report reconstructed only after rereading those tables.
- Make model-inspection label provenance explicit: the real-data post-analysis examples read
  predictor and response names from `X.csv` and `Y.csv` headers, while plotting remains independent
  of pandas and file layout and accepts labels from any caller-owned metadata source.
- Present selected Pi-PLS and ordinary PLS components together on shared axes, using side-by-side bars for named categorical variables and overlaid lines for physical predictor axes; require caller-supplied scientific labels for categorical predictor and response displays.
- Add immutable ordinary PLS latent-structure extraction from public `PLSRegression` scores,
  X/Y loadings, and coefficients, together with optional score, loading, and response-specific
  coefficient figures used by the maintained real-data post-analysis workflows.
- Add optional `pipls.plotting` figures for Pi-PLS $P$, $D$, and $QD$ displays and
  explicit-provenance prediction diagnostics.
- Add the pure `pipls.inspection` foundation with immutable Pi-PLS display factors,
  deterministic sign canonicalization that preserves $PDQ^\mathsf{T}$, and explicit-provenance
  prediction diagnostics standardized from observed responses with `ddof=1`.
- Accept the staged model-inspection and post-analysis architecture: separate component-path
  diagnostics, fitted-model interpretation, and prediction diagnostics; reserve pure numerical
  work for `pipls.inspection`, optional figures for `pipls.plotting`, and dataset-specific artifact
  orchestration for `examples/`.
- Complete the legacy real-dataset licensing review: retain pulp, sugarcane, and tobacco as the
  repository suite; intentionally exclude Corn, the legacy Citrination Steel table, SARCOS, and
  FRED-MD because the exact source materials do not carry sufficiently clear redistribution rights;
  and return the roadmap to documentation and release hardening.
- Synchronize the guide layer with the final path API: describe score-maximizing selection and
  adaptive-search limits accurately, clarify custom-scorer interpretation of component-path MSE,
  correct the Pulp example narrative, document the centered direct-fit rank limit, and include the
  user guides in source distributions.
- Complete final scikit-learn API polish: make `n_components_values="all"` the explicit complete-path default, accept integer/NumPy `RandomState`/`None` random-state forms, use the public response-standardized scorer callable as the path default, hide refit-dependent path methods when unavailable, and remove duplicate Pi-PLS fitted aliases in favor of canonical `decomposition_` fields.
- Complete the fixed-estimator/path-search correction with a final API and minimality audit: remove the redundant `pipls_param_prefix` control, infer the terminal Pi-PLS pipeline step, and protect fixed-pair `GridSearchCV` interoperability without recommending it in examples.
- Simplify the real-data examples around the final two-stage workflow: import the PLS-path and CSV-to-PDF helpers directly, remove subprocess wrappers and repeated table-validation boilerplate, and fit the fixed final model from the chosen canonical Pi-PLS CSV row.
- Remove obsolete private selection machinery after the fixed-estimator split: delete unused rank-grid and solver-tracing hooks, eliminate duplicate candidate metadata, move response-standardized loss primitives to `metrics.py`, and generate OOF predictions without rescoring the selected candidate.
- Make `PiPLSPathCV` the sole package selection owner, including explicit control of support-warning suppression across feature probes, candidate folds, optional OOF fits, and the selected full-data refit; unrelated warnings continue to propagate.
- Make `PiPLSRegression` a direct fixed-pair estimator with no embedded CV or search results; add the direct-fit support warning at fewer than three observations per retained predictor direction while preserving `PiPLSPathCV` candidate behavior.

- Record the staged fixed-estimator/path-search boundary, exclude generated example artifacts from snapshots, repair malformed documentation LaTeX, and correct stale core/preprocessing decision statuses.
- Add a single `make examples` application-validation target that runs every numbered example in
  order, including the complete Tobacco analysis, while keeping `make check` focused on fast
  internal contracts.
- Remove the duplicate Pulp, Sugarcane, and Tobacco smoke benchmark scripts and tests, and stop
  executing complete real-data examples in `make check`; retain fast dataset, path, PLS-helper,
  plotting, and workflow-structure contracts.
- Compare the Pulp, Sugarcane, and Tobacco Pi-PLS component paths with separate standard PLS (NIPALS) CSV paths and CSV-derived overlaid PDFs; use explicit full predictor SVD for the Tobacco path while retaining adaptive predictor-rank scanning.
- Add a dedicated `examples` dependency extra containing pandas and Matplotlib, and document how to refresh an existing development virtual environment after dependency changes.
- Present real-data model development as a two-stage component-path workflow: add the public
  `component_path_results_` table, explicit optimized/fixed/maximum predictor-rank policies,
  four-row Pulp and Sugarcane benchmark CSVs with fold SD, CSV-derived example PDFs, and separate
  fixed final-model fits chosen through visible component-count constants.
- Add the Sugarcane high-dimensional component-path smoke check using direct pandas tables and the ordinary public `PiPLSPathCV` defaults.

- Remove the Linnerud dataset integration, its executable example, and dataset-specific test
  because it does not provide a useful representative Pi-PLS workflow; retain pulp, sugarcane,
  and tobacco as the transparent real-data suite.
- Derive the rule-based predictor-rank support term from the total number of observations supplied
  to `fit()`, while retaining centered training-fold dimensions as hard feasibility caps; keep the
  public defaults at `samples_per_predictor_rank=5` and `cv=5` and remove the temporary Pulp
  `samples_per_predictor_rank=4` override.
- Set the public rank-selection defaults to `samples_per_predictor_rank=5` and `cv=5` in both
  `PiPLSRegression` and `PiPLSPathCV`, and simplify the Pulp example and smoke check to use the
  ordinary rule-derived predictor-rank bound without an explicit maximum.
- Add the transparent Pulp component-path smoke check using direct pandas `X.csv`/`Y.csv` reading and the ordinary public `PiPLSPathCV` workflow.
- Add the full-versus-randomized solver-consistency benchmark with paired fixed ranks, three high-dimensional matrix geometries, independent-test prediction and coefficient relative differences, a minimal four-column CSV, and focused contract tests.
- Add the paired predictor-nuisance benchmark comparing fixed Pi-PLS and ordinary PLS on identical deterministic synthetic train/test problems, with controlled nuisance strengths, training-fitted model standardization, independent-test MSE, a minimal five-column CSV, and focused contract tests.
- Add the adaptive rank-selection benchmark with deterministic public synthetic train/test data, generator-declared reference ranks, fold-local model standardization, full-training refit, independent-test MSE, a minimal seven-column CSV output, and focused repeatability tests.
- Add the fixed-structure recovery benchmark with deterministic public synthetic train/test data, fixed oracle Pi-PLS ranks, test MSE, coordinate-correct subspace captures, a minimal six-column CSV output, and focused repeatability tests.
- Replace the over-general synthetic benchmark manifest, universal schema, and broad CI runner with a focused benchmark plan: one user question and one minimal CSV output per benchmark.

- Remove paper-reproduction placeholders and reorganize public navigation around the installable package, user documentation, concise examples, transparent datasets, and lightweight validation responsibilities.

- Add the CC BY 4.0 tobacco FT-NIR dataset as the fourth transparent real-data integration, with public sample-ID alignment, 1,557 raw spectral predictors, 13 chemical responses, no spectral preprocessing, and direct pandas I/O.

- Add the CC BY 4.0 sugarcane LabSpec dataset as the third transparent real-data integration, with explicit sample alignment, missing-response exclusion, wavelength selection, compact spectral-axis metadata, and direct pandas I/O.

- Replace brittle repository-document and dataset-metadata value assertions with structural file, format, decision-index, and executable-example tests; document the durable testing policy.

- Restrict committed dataset provenance to public or included sources, remove private pulp archive references and its preparation-only converter, and initially reserve Corn as a user-facing raw-data preprocessing exception; Decision 0041 later excludes it after licensing review.

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
