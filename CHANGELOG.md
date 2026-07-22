# Changelog

## Unreleased

- Complete the pre-release result and example simplification: remove the duplicate matrix-shaped
  score and response-standardized-MSE path attributes, retain `cv_results_` as the sole detailed
  candidate surface, align documentation with direct in-memory workflows, and add structural tests
  that prevent generated analytical CSV intermediates or file-based plotting in numbered examples.

- Make the Pi-PLS/ordinary-PLS path comparison direct: return immutable ordinary-PLS path arrays,
  plot both methods in example 09 with ordinary Matplotlib, write only three final PDFs, and remove
  the comparison CSV intermediates and plotting helper.

- Make Tobacco a direct in-memory workflow: preserve full predictor SVD, adaptive rank scanning,
  decreasing-wavenumber plots, deterministic source-order response pagination, and raw observation
  diagnostics; write five final PDFs, with three-page prediction and coefficient files, and remove
  the fixed-model OOF and post-analysis CSV/report helpers.

- Make the Pulp example and tutorial direct: remove the one-step workflow wrapper and generated
  analysis CSV round trips, calculate path selection, fixed fitting, scikit-learn OOF predictions,
  and immutable inspection results visibly in example 10, add a three-component predictor-rank
  profile, and write six final PDF figures from in-memory results.

- Make Sugarcane the direct reference workflow: plot `component_path_` in memory, use
  scikit-learn `cross_val_predict()` for the selected fixed model, pass immutable inspection results
  directly to atomic plotters, and write five final PDF figures without generated analytical CSV
  intermediates.

- Complete the tutorial-first documentation transition: make the Pulp tutorial the sole worked analysis, shorten task guides, separate the path-search reference from the selection how-to, and turn model inspection into the stable figure-by-figure interpretation reference.

- Add a tutorial-first documentation route built around the canonical Pulp workflow: include checked source snippets, all generated figures one chart at a time, explicit selection and validation provenance, and links to the general inspection, plotting, and theory references.

- Generate deterministic single-chart SVG assets and a machine-readable manifest for the planned Pulp tutorial from the canonical workflow; integrate generation into documentation builds and clean source-distribution validation.

- Establish one canonical Pulp workflow for the numbered example and planned tutorial: evaluate a terminal-Pi-PLS pipeline, transfer the selected nested rank pair to a fixed clone, generate selection-conditioned OOF predictions, and compute the shared inspection results without duplicating the numerical analysis.

- Complete the plotting-composition audit: make the example report layer create every figure and
  axis, group shared latent-model views into dataset-appropriate panels, retain full-width
  coefficient pages, and enforce structurally that package plotters remain one-axis primitives.

- Replace the composite prediction-diagnostics figure with separate one-axis plots for observed
  versus predicted responses, residuals versus predicted responses, and standardized RMSE; move
  prediction-panel composition, provenance, legends, PDF writing, and closing into the example
  layer.

- Replace the composite Pi-PLS decomposition figure with separate one-axis plots for $P$, $D$,
  $Q$, and $QD$; move factor-panel composition, legends, titles, PDF writing, and closing into the
  example layer.

- Establish the single-axis plotting contract for the existing atomic PLS-family figures: accept
  caller-supplied Matplotlib axes, return `(figure, axis)`, leave panel composition and legends to
  the caller, and retain standalone one-axis figure creation.

- Refine the public documentation entry: exclude maintainer decision records from the served site,
  introduce Pi-PLS through paired latent variables and CV-MSE component scanning, link introductory
  material to the theory guide, and shorten the compatibility page.

- Add clean installed-distribution validation: build the wheel and source distribution once,
  install each into a separate temporary environment outside the checkout, and run one shared
  public-import, metadata, fit, prediction, and import-origin smoke test in CI.
- Separate compatibility CI into diagnosable minimum-dependency, supported-Python, and
  latest-compatible jobs; print the resolved Python, NumPy, scikit-learn, and joblib versions in
  every environment.
- Define the first-release compatibility policy for Python 3.10–3.14, NumPy 1.26--2.x,
  scikit-learn 1.4--1.x, and joblib 1.2--1.x; add minimum-dependency constraints, Python
  classifiers, guarded runtime ranges, Python 3.14 CI coverage, and consistency tests.
- Add a self-documenting Make interface: `make` and `make help` list the maintained targets, and
  `make docs-serve` provides a memorable live documentation preview at
  `http://127.0.0.1:8000/`.
- Validate the strict documentation site in CI from both the repository checkout and a clean
  installation of the unpacked source distribution; ship the MkDocs configuration, Makefile, and
  validation helper; and exclude generated `site/` output from repository snapshots.
- Complete the generated public API reference for inspection, plotting, datasets, and metrics;
  audit immutable-result shapes, plotting contracts, synthetic generators, and scorer semantics;
  formalize the dataset submodule exports; and verify that importing plotting remains independent
  of Matplotlib.
- Preserve the tracked example-result directory placeholders in repository snapshots while still
  excluding generated analysis artifacts.
- Add generated API reference pages for the fixed estimator, path selector, decomposition,
  validation report, and support warning; introduce mkdocstrings with Ruff-formatted signatures,
  and audit the corresponding public docstrings for parameters, fitted attributes, shapes, and
  conditional outputs.
- Add a strict MkDocs documentation build with Material navigation, MathJax rendering, dedicated documentation dependencies, and ignored `site/` output.
- Define `docs/` as the self-contained public documentation source, add public example and design-decision navigation, expand the implemented Pi-PLS theory guide, and remove unused selection alternatives from current documentation.
- Align component-path documentation with the current example ownership: example 09 owns explicit
  Pi-PLS-versus-PLS comparisons, examples 10–12 use Pi-PLS-only paths, fold SD remains descriptive,
  and component count remains an explicit path-based choice.
- Remove the context-free advanced-cross-validation example; keep grouped, leave-one-out, and temporal splitters in the dedicated cross-validation documentation.
- Rewrite the synthetic-data example as an explained independent train/test use case with labeled matrix dimensions, latent structure, fitted parameters, prediction shape, and test $R^2$.
- Require numbered examples to demonstrate self-contained user tasks, comparisons, or benchmarks without publication context.
- Separate comparison from normal analysis examples: add one dedicated Pulp, Sugarcane, and Tobacco Pi-PLS-versus-PLS path-comparison example, and make examples 10–12 write Pi-PLS-only `component_path.csv` and `component_path.pdf` beside each dataset post-analysis report.

- Shorten the numbered examples without changing their analyses: inline one-use arguments, remove redundant checks of committed CSV headers and axis order, use one selected Pi-PLS object for fixed-fit and OOF calls, reduce console scaffolding, and track required result directories instead of creating them at runtime.

- Complete the Decision 0045 analysis-boundary migration: enforce ordinary PLS as a comparison-only
  model in numbered examples, keep shared PLS-family inspection estimator-neutral, use one selected
  Pi-PLS model for post-analysis, and align quantity-based artifact and documentation contracts.

- Convert the Pulp, Sugarcane, and Tobacco post-analysis workflows to one selected Pi-PLS model
  after the retained Pi-PLS-versus-PLS CV-MSE comparison; derive all shared analyses from that
  Pi-PLS fit, use estimator-neutral artifact filenames, and remove stale `pls_*.csv` files during
  regeneration.
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
  `component_path_` immutable result, explicit optimized/fixed/maximum predictor-rank policies,
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
