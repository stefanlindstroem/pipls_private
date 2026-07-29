# Decision-record navigation

This file is a compact index for LLM-assisted work. The records under `docs/decisions/` contain the
accepted rationale and consequences. Read the full record whenever a change touches its subject.
This index is navigation, not a substitute for those records.

| Record | Subject | Consequence |
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
| `0010-path-analysis-api.md` | triangular path search | pipeline-aware `PiPLSSearchCV` with aligned search vocabulary |
| `0011-shared-selection-engine.md` | code ownership | path-owned private fold/search machinery supports fixed candidate evaluation |
| `0012-sklearn-api-alignment.md` | estimator and PLS compatibility | standard fitted surface plus structured Pi-PLS decomposition output |
| `0013-sklearn-cleanup-boundary.md` | final pre-D2 scope | direct estimator or terminal-pipeline support and conditional delegation |
| `0014-validation-metadata-scope.md` | groups and weighting boundary | groups-only splitter metadata; no weighted fitting or general routing |
| `0015-dataset-and-synthetic-api.md` | dataset and synthetic boundary | optional immutable datasets plus local seeded latent-structure generation |
| `0016-transparent-data-ingestion.md` | real-data and example boundary | users and examples read `X` and `Y` explicitly; no required registry or generic loader |
| `0017-first-real-dataset.md` | former first transparent real-data integration | superseded by Decision 0033; the Linnerud integration is removed |
| `0018-repository-dataset-layout.md` | committed real-dataset file convention | every dataset uses comma-delimited `X.csv`, `Y.csv`, and documentary `metadata.yaml` |
| `0019-pulp-dataset-integration.md` | pulp dataset integration | pulp uses public supplementary provenance, CC BY 4.0 attribution, named column selection, and direct X/Y reading |
| `0020-public-dataset-provenance-boundary.md` | public versus internal dataset materials | public-only provenance and no private paths or preparation-only scripts; Corn-specific plan superseded by 0041 |
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
| `0037-user-run-real-data-analyses.md` | real-data execution boundary | examples remain user-run; duplicate smoke benchmarks and full example tests are removed |
| `0038-single-examples-target.md` | explicit application validation | `make examples` runs every numbered example; `make check` remains fast internal validation |
| `0039-fixed-estimator-path-search-boundary.md` | estimator versus selection ownership | implemented split: fixed `PiPLSRegression`, triangular selection in `PiPLSSearchCV` |
| `0040-sklearn-api-polish.md` | final public API polish | explicit complete-path sentinel, conventional random state, callable scorer, and canonical decomposition output |
| `0041-legacy-dataset-licensing-roadmap.md` | legacy dataset licensing and roadmap | retain the three licensed datasets; exclude Corn, legacy Steel, SARCOS, and FRED-MD |
| `0042-model-inspection-and-post-analysis.md` | fitted-model analysis architecture | separate selection diagnostics, interpretation, and prediction diagnostics; reusable inspection, plotting, and all three real-data integrations |
| `0043-pls-biplot-and-analysis-surface.md` | final fitted-model analysis increment | balanced reconstruction-preserving Pulp biplot; no spectral biplots; analysis series complete |
| `0044-minimal-onboarding-and-example-support.md` | user onboarding and example organization | literal-matrix first example; complete-workflow helpers under `examples/_support/`; one examples target retained |
| `0045-pls-family-analysis-boundary.md` | comparison versus fitted-model analysis ownership | retain ordinary PLS for CV-MSE comparison; keep $P$, $D$, and $Q$ Pi-PLS-specific; make shared analysis estimator-neutral and apply it only to Pi-PLS in numbered examples |
| `0046-concise-numbered-examples.md` | pedagogical example minimality | keep scientific stages explicit; remove one-use scaffolding and redundant repository-data checks; track required result directories |
| `0047-separate-pls-path-comparison-example.md` | comparison versus normal analysis examples | isolate all PLS path comparisons in the dedicated comparison example; Decision 0101 later renumbers it to example 04 and the Pi-PLS-only real-data analyses to examples 05–07 |
| `0048-self-contained-user-examples.md` | user-facing example semantics | remove context-free advanced CV; require explained data, labeled output, and publication-independent numbered examples |
| `0049-public-documentation-source-boundary.md` | public documentation ownership | `docs/` is self-contained; user guides describe implemented behavior and do not depend on `.llm` or repository-external Markdown |
| `0050-buildable-documentation-foundation.md` | strict MkDocs foundation | Material, MathJax, user navigation, and `make docs` define the buildable public site |
| `0051-core-generated-api-reference.md` | core generated API documentation | mkdocstrings pages cover the supported top-level objects and audit parameters, fitted attributes, shapes, and conditional outputs |
| `0052-complete-generated-api-reference.md` | complete generated API documentation | inspection, plotting, datasets, and metrics are generated from audited public docstrings with explicit submodule coverage |
| `0053-documentation-distribution-validation.md` | documentation artifact validation | CI builds the strict site from the checkout and from a clean installation of the unpacked source distribution |
| `0054-compatibility-policy.md` | supported interpreter and dependency ranges | Python 3.10–3.14, guarded runtime dependency majors, and one Python 3.10 minimum stack |
| `0055-compatibility-ci-matrix.md` | executable compatibility environments | separate minimum, supported-Python, and latest-compatible jobs with resolved-version diagnostics |
| `0056-installed-distribution-validation.md` | installed release-artifact validation | build once, then verify clean wheel and source-distribution installations with one shared public smoke test |
| `0057-public-documentation-entry.md` | public documentation entry and maintainer-record boundary | exclude decisions from the served site, define latent-variable selection before jargon, and keep compatibility concise |
| `0058-single-axis-plotting-contract.md` | reusable plotting composition | one chart per public plotter, optional caller-supplied axes, caller-owned legends and panel layout |
| `0059-atomic-pipls-factor-plots.md` | Pi-PLS factor plotting surface | separate one-axis $P$, $D$, $Q$, and $QD$ plots; caller-owned factor panels |
| `0060-atomic-prediction-diagnostic-plots.md` | prediction-diagnostic plotting surface | separate one-axis observed/predicted, residual, and standardized-RMSE plots; caller-owned diagnostic panels |
| `0061-example-owned-report-composition.md` | complete example plotting ownership | example reports create every figure and axis, group related charts in dataset-appropriate panels, and leave package plotters atomic |
| `0062-canonical-pulp-workflow.md` | shared Pulp tutorial analysis | one example-owned pipeline workflow supplies path evaluation, fixed fitting, OOF prediction, and inspection to the numbered example and future tutorial assets |
| `0063-repository-generated-pulp-tutorial-figures.md` | generated Pulp tutorial assets | one deterministic SVG per chart plus a manifest, regenerated by documentation targets from the canonical workflow |
| `0064-tutorial-first-documentation.md` | primary pedagogical route | one complete Pulp tutorial uses executable snippets and generated figures, with guide and API links for deeper reference |
| `0065-documentation-layer-consolidation.md` | tutorial, guide, and reference ownership | tutorial owns the worked Pulp analysis; guides own procedures; model inspection and generated API pages own general interpretation and signatures |
| `0066-immutable-component-path-api.md` | concise path-result API | frozen aligned path arrays, scalar component lookup, and removal of redundant fitted mappings |
| `0067-direct-sugarcane-workflow.md` | direct Sugarcane reference workflow | in-memory path, OOF, inspection, and explicit final-PDF figure composition without generated analytical CSV intermediates |
| `0068-direct-pulp-workflow.md` | direct Pulp example and tutorial | visible direct path, rank-profile, fixed-fit, OOF, inspection, and final-figure workflow without a shared wrapper or analytical CSV intermediates |
| `0069-direct-tobacco-workflow.md` | direct Tobacco example | visible full-SVD path, fixed-fit, OOF, inspection, response pagination, and final-PDF workflow without analytical CSV intermediates or report helpers |
| `0070-direct-pls-path-comparison.md` | direct Pi-PLS/PLS comparison | immutable ordinary-PLS path arrays, direct example-owned Matplotlib composition, and removal of comparison CSV intermediates and plotting helper |
| `0071-final-result-and-example-cleanup.md` | final result and example cleanup | remove duplicate matrix-path aliases, make `cv_results_` the sole detailed surface, and enforce direct in-memory numbered examples |
| `0072-conditional-predictor-rank-profile.md` | conditional predictor-rank inspection | derive one immutable sorted rank profile on demand from `cv_results_` without another fitted representation |
| `0073-public-fit-safety.md` | public fit-state and finite-output safety | transactional fits, boundary-safe scaling, safe `copy=False`, and explicit rejection of nonfinite public results |
| `0074-tutorial-owned-user-workflow.md` | consolidated user documentation | tutorial owns normal use, generated API pages own exact estimator contracts, and advanced guides retain only specialized behavior |
| `0075-two-tier-tutorial-route.md` | staged programming-user tutorials | synthetic selection and prediction first; complete Pulp analysis second |
| `0076-focused-pulp-tutorial.md` | focused second-stage tutorial | real-data selection qualification, selection-conditioned OOF analysis, and representative fitted-model plots |
| `0077-audience-oriented-documentation-entry.md` | user and maintainer documentation entry | concise user README, contributor-owned maintenance commands, and project-validation navigation |
| `0078-documentation-reference-cleanup.md` | documentation reference cleanup | result-object map, troubleshooting, generic link and anchor checks, and tests that avoid freezing living prose |
| `0079-data-first-biplot-rendering.md` | data-first biplot rendering | retain balanced coordinates, remove `plot_biplot()`, and use direct Matplotlib with optional `adjustText` label layout |
| `0080-direct-prediction-diagnostic-rendering.md` | direct prediction-diagnostic rendering | immutable diagnostic arrays are primary; examples use direct Matplotlib; three convenience plotters removed |
| `0081-direct-standard-inspection-rendering.md` | direct standard inspection rendering | immutable latent and observation arrays are primary; examples use direct Matplotlib; five convenience plotters removed |
| `0082-direct-pipls-factor-rendering.md` | direct Pi-PLS factor rendering | immutable factor arrays are primary; plotting module and plot extra removed |
| `0083-data-first-rendering-policy.md` | final data-first rendering policy | immutable results are the compatibility surface; rendering remains optional and caller-owned |
| `0084-complete-pulp-prediction-diagnostics.md` | complete Pulp prediction diagnostics | Tutorial 2 displays the residual figure represented by the middle axis of its maintained three-panel prediction snippet |
| `0085-self-contained-pulp-tutorial.md` | self-contained Pulp tutorial | setup and display choices precede use; each shown figure has a matching standalone recipe |
| `0086-public-decomposition-boundary.md` | public decomposition boundary | public result exposes final rotations, dilation, rank/solver diagnostics, and standardized map; construction matrices remain private |
| `0087-public-fitted-surface-cleanup.md` | public fitted-surface cleanup | remove scorer plumbing, exact rotation aliases, path execution bookkeeping, and flat OOF duplicates |
| `0088-public-result-record-cleanup.md` | public result-record cleanup | remove display-sign bookkeeping and impossible zero loading blocks; hide constructors for returned records |
| `0089-balanced-pulp-factorization-views.md` | balanced Pulp factorization views | Tutorial 2 displays both $P$ and $QD$; the complete example retains the separate $D$ and $Q$ plots |
| `0090-reference-consolidation.md` | programming-reference consolidation | merge validation with path details, colocate small generated API groups, and shorten inspection prose while preserving public objects and anchors |
| `0091-clean-git-snapshots.md` | clean committed-tree snapshots | refuse tracked, staged, or nonignored untracked changes and archive `HEAD` so ignored local files cannot enter handoffs |
| `0092-fold-numerical-rank-feasibility.md` | fold numerical-rank feasibility | cap path candidates by the minimum rank verified after fold-local preprocessing before scoring |
| `0093-public-result-invariants.md` | immutable core public-result invariants | validate direct construction, defensive copies, scalar normalization, OOF coverage, and pickle reconstruction |
| `0094-inspection-result-safety.md` | immutable and finite inspection results | validate direct construction and pickle reconstruction; use range-safe calculations and reject unrepresentable derived values |
| `0095-required-fixed-rank-pair.md` | required fixed Pi-PLS rank pair | require keyword-only `n_components` and `predictor_rank`; path templates use a replaceable valid seed pair |
| `0096-notation-path-ceilings-and-inspection-navigation.md` | notation, path ceilings, and inspection navigation | distinguish mathematical $Y$ from scikit-learn `y`, define both path ceilings first, and group inspection concepts with its API |
| `0097-focused-leave-one-out-example.md` | focused leave-one-out example | show one small calibration workflow with singleton-safe scoring, ordered OOF predictions, and pooled OOF $R^2$ |
| `0098-grouped-make-help.md` | grouped maintainer command index | highlight setup and routine validation, then group unchanged Make targets by task |
| `0099-example-artifact-and-sdist-boundary.md` | example artifact and source-distribution boundary | commit and snapshot only result-directory placeholders; include them in sdists and run example 01 from an extracted sdist |
| `0100-rendered-documentation-deployment.md` | rendered documentation deployment | validate strict docs on pushes and pull requests; deploy the canonical repository-derived GitHub Pages site from `master` |
| `0101-continuous-numbered-examples.md` | continuous numbered examples | renumber maintained examples 01--07, migrate active consumers, and label example 01 terminal output |
| `0102-path-search-defaults.md` | path-search defaults and scorer presentation | default to selection-only `refit=False`; use a stable package scorer name resolving to the public callable |
| `0103-installation-and-optional-dependency-boundary.md` | installation and optional dependencies | retain only maintained `dev`, `examples`, and `docs` extras; use noneditable public installation and editable contributor setup |
| `0104-new-user-documentation-route.md` | new-user documentation route | motivate the separate ranks without performance claims and defer tutorial maintenance details to reproduction sections |
| `0105-documentation-implementation-alignment.md` | documentation and implementation alignment | demonstrate current path defaults and document complete decomposition, output, and adaptive-search surfaces |
| `0106-fold-based-cv-standard-error.md` | fold-based CV standard error | derive read-only standard errors from stored population fold SD and split counts without changing selection or plots |
| `0107-component-path-recommendation-methods.md` | component-path recommendation methods | return exact stored minimum-CV-MSE and one-standard-error rows without fitting, mutation, tolerances, or redundant state |
| `0108-tobacco-one-standard-error-workflow.md` | Tobacco one-standard-error workflow | use the stored 1-SE recommendation for example 07 while keeping introductory workflows explicit |
| `0109-tobacco-one-standard-error-threshold-figure.md` | Tobacco one-standard-error threshold figure | show the minimum row, horizontal 1-SE threshold, and recommended row with direct documentation cross-links |
| `0110-response-anchored-display-factors.md` | response-anchored Pi-PLS display factors | retain predictor-canonical defaults; optionally orient every component by a selected response row and requested sign |
| `0111-explicit-path-selection-rules.md` | explicit path selection rules | preserve global `best_*`; expose a separate selected path row and optionally refit the best-score or 1-SE choice |
| `0112-search-cv-public-name.md` | public selection-class name | rename the unreleased meta-estimator to `PiPLSSearchCV` without an alias; retain component-path result terminology |
| `0113-derived-weighted-response-directions.md` | derived weighted response directions | store only independent display factors and expose checked read-only $QD$ as a derived property |
| `0114-derived-prediction-diagnostics.md` | derived prediction diagnostics | accept observed values, predicted values, and provenance; derive all dependent diagnostic arrays once |
| `0115-normalized-path-result-state.md` | normalized path result state | store path-wide policy and split count once; derive predictor-rank-profile selection from candidate arrays |
| `0116-composed-validation-report-result.md` | composed validation-report result | store one selected component result and derive the existing selected-row convenience attributes |
| `0117-commercial-license-authorship-and-citation.md` | commercial license, authorship, and citation | retain complete BSD-3-Clause terms, name the three copyright holders, and publish software plus companion-paper citation metadata |
| `0118-maintained-figure-labeling-and-axis-policy.md` | maintained figure labeling and axis policy | consistent $\Pi$-PLS notation, factor symbols, numeric dilation ticks, tile titles, path/profile scales, and Tobacco response labels |
| `0119-manuscript-latent-geometry-generator.md` | manuscript latent-geometry generator | additive exact Gaussian manuscript generator with manuscript-oriented immutable truth; existing synthetic and real-data workflows unchanged |
| `0120-companion-manuscript-theory-alignment.md` | companion-manuscript theory alignment | canonical projector/optimization/diagonal derivation, corrected fitted dimension, and explicit manuscript/package scope boundary |

## Implemented estimator/search transition

Decisions 0039 and 0040 are fully implemented. `PiPLSRegression` is a fixed-pair estimator with
the direct-fit support warning at $n/r_\pi<3$. `PiPLSSearchCV` owns the complete triangular-selection
lifecycle and defaults to the explicit complete-component sentinel `n_components_values="all"`.
Decision 0112 renames the sole package selection meta-estimator to `PiPLSSearchCV` without a
legacy alias; component-path result names and numerical behavior are unchanged. Decision 0102 makes path evaluation selection-only by default and represents the default scorer by
a stable package string resolving to the existing public callable. Conventional scikit-learn
random-state forms remain accepted, and Pi-PLS-specific fitted output is canonicalized in
`decomposition_`.

## Accepted clarifications after earlier proposals

These points are fixed by implemented decisions and owner review even where the broad publication
plan contains an earlier or more general proposal:

- both adaptive public defaults use the name `"auto"`; exhaustive search is explicit `"optimal"`;
- `PiPLSSearchCV` defaults to `n_components_values="all"`; explicit sequences request a subset;
- `PiPLSSearchCV` defaults to selection-only `refit=False`; explicit `selection_rule` chooses the
  best-score or stored 1-SE row for optional final refitting;
- the default scoring parameter is the stable package name `"neg_response_standardized_mean_squared_error"`, which resolves to the public scorer callable;
- `PiPLSSearchCV` defaults to `samples_per_predictor_rank=5` and `cv=5`;
- the samples-per-rank support term uses the total number of observations supplied to `fit()`,
  while transformed feature count, centered training-fold dimensions, and minimum verified
  fold numerical rank remain hard candidate-feasibility caps;
- randomized SVD is controlled independently and follows the same policy inside regression and
  path candidate fits;
- `PiPLSRegression` fits explicit ranks only; `PiPLSSearchCV` owns package selection and fits fixed
  estimator clones;
- `PiPLSSearchCV` supports a direct estimator or a scikit-learn `Pipeline` ending in
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
- the first numbered example uses literal NumPy matrices and one fixed fit; complete-workflow
  helpers are separated under `examples/_support/`;
- numbered examples trust committed dataset and result-directory structure, avoid one-use configuration constants and repeated validation scaffolding, and leave reusable contract validation to focused tests;
- numbered examples are self-contained user tasks with explained data and labeled output; they do not rely on paper or manuscript context, and the context-free advanced-CV script is removed;
- committed repository datasets use comma-delimited `X.csv`, `Y.csv`, and `metadata.yaml`, while
  external users remain free to use any data source or file organization;
- the current real-data integration suite is pulp, sugarcane, and tobacco; all are repository
  example data, not runtime loaders or publication-result claims;
- committed dataset assets use public or included provenance only, and the exact source material
  must carry an explicit redistribution and adaptation grant; private archive references and
  preparation-only scripts stay outside the repository;
- Corn, the legacy Citrination Steel table, SARCOS, and FRED-MD are intentionally excluded under
  Decision 0041; separately licensed derivatives are new candidate datasets, not retroactive
  clearance of the companion-analysis files;
- fitted-model analysis is separated into selection diagnostics, full-data interpretation, and
  prediction diagnostics under Decision 0042; numerical inspection is package-owned and
  dataset-specific artifacts remain example-owned; Decisions 0079--0083 supersede the former
  package plotting layer with direct caller-owned rendering;
- Decision 0045 further separates model roles: ordinary PLS remains in component-path comparisons
  and declared comparator benchmarks; $P$, $D$, and $Q$ inspection stays Pi-PLS-specific; shared
  score, loading, coefficient, biplot, observation, and prediction analysis uses estimator-neutral
  APIs and is applied only to the selected Pi-PLS model in numbered examples;
- Decision 0047 isolates all real-data Pi-PLS-versus-PLS path comparisons in the dedicated
  comparison example; Decision 0101 assigns it the current number 04, while examples 05–07 are the
  direct Pi-PLS-only Pulp, Sugarcane, and Tobacco workflows.
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
- example 04 plots immutable Pi-PLS and ordinary-PLS component paths directly in memory; Pulp,
  Sugarcane, and Tobacco examples 05–07 plot `component_path_` directly in memory; Pulp and
  Sugarcane fit separate fixed models after explicit component-count choices, while Tobacco calls
  the stored 1-SE recommendation explicitly and fits the returned component-count/predictor-rank
  pair;
- fold SD remains stored descriptive dispersion; maintained CV-MSE figures use the derived
  fold-based standard error. Maintained examples keep component choice visible, while an explicit
  `selection_rule` may automate a protocol declared before fitting;
- example 04 compares separate immutable Pi-PLS and standard PLS component paths in one PDF per
  dataset; examples 05–07 write final PDFs directly from in-memory Pi-PLS results, with Pulp exposing
  the conditional predictor-rank profile and Tobacco preserving full-SVD spectral plots,
  source-order response pagination, and raw observation diagnostics; randomized SVD is covered by
  the solver-consistency benchmark;
- Python 3.10–3.14 are supported; runtime metadata admits NumPy 1.26--2.x,
  scikit-learn 1.4--1.x, and joblib 1.2--1.x; the complete minimum stack is tested only on
  Python 3.10 because its oldest binary releases do not cover every newer interpreter;
- complete Pulp, Sugarcane, and Tobacco analyses are not duplicated as real-data benchmark scripts
  or executed by the default test suite; `make examples` runs all numbered examples explicitly;
- the package exposes no public plotting module; maintained reports render immutable inspection
  arrays directly with Matplotlib, group related charts in dataset-appropriate panels, and retain
  full-width coefficient pages;

When a new accepted architectural or public-API decision is added, create a numbered decision
record and add it to this index in the same patch.
