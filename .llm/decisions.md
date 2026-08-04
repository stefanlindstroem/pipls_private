# Decision-record navigation

This file is the machine-checkable registry of numbered decisions currently shipped under
`docs/decisions/`. The full records contain rationale and consequences; read the relevant record
before changing its subject. Table descriptions summarize the original record and may describe an
intermediate state later superseded by a newer decision.

Decision 0147 governs consolidation. This registry lists only numbered decisions currently shipped
under `docs/decisions/`. Retired filenames and canonical replacements are recorded in
`docs/decisions/retirements.md`; Git remains the archive and decision numbers are never reused. A
compact development-era summary will be added in Patch 4.

| Record | Subject | Original durable consequence or current disposition |
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
| `0018-repository-dataset-layout.md` | historical repository-only dataset convention | retained for future repository-only assets; current named reference datasets follow Decision 0142 package resources |
| `0019-pulp-dataset-integration.md` | pulp dataset integration | pulp uses public supplementary provenance, CC BY 4.0 attribution, named column selection, and direct X/Y reading |
| `0020-public-dataset-provenance-boundary.md` | public versus internal dataset materials | public-only provenance and no private paths or preparation-only scripts; Corn-specific plan superseded by 0041 |
| `0021-durable-repository-tests.md` | repository test stability | test behavior and structural format without pinning living prose or documentary field values |
| `0022-sugarcane-dataset-integration.md` | sugarcane dataset integration | public LabSpec spectra, explicit row and wavelength selection, and compact regular-axis metadata |
| `0023-tobacco-dataset-integration.md` | tobacco dataset integration | public raw FT-NIR spectra, explicit ID alignment, metadata-column exclusion, and no spectral preprocessing |
| `0024-package-product-repository-boundary.md` | package versus publication ownership | `pipls` owns the software product; paper reproduction stays downstream; future block-aware API design is deferred |
| `0025-model-internal-standardization-boundary.md` | current versus future scaling | estimator centering/scaling is current and fold-local; only future block-aware variants are deferred |
| `0026-package-navigation-cleanup.md` | public repository navigation | remove paper placeholders and organize entry points around the installable software product |
| `0031-default-selection-support.md` | ordinary rank-selection defaults | five samples per retained predictor direction and five-fold CV by default |
| `0032-full-sample-rank-support.md` | rank-support sample-count convention | full supplied $n$ defines support; centered training folds impose feasibility caps |
| `0033-remove-linnerud-integration.md` | reference dataset scope | remove the Linnerud dataset, example, test, and active navigation |
| `0034-two-stage-component-path-workflow.md` | component-path presentation | one CSV row per component count, explicit predictor-rank policy, CSV-derived PDF, and separate fixed final fit |
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
| `0110-response-anchored-display-factors.md` | response-anchored Pi-PLS display factors | retain predictor-canonical defaults; optionally orient every component by a selected response row and requested sign |
| `0112-search-cv-public-name.md` | public selection-class name | rename the unreleased meta-estimator to `PiPLSSearchCV` without an alias; retain component-path result terminology |
| `0113-derived-weighted-response-directions.md` | derived weighted response directions | store only independent display factors and expose checked read-only $QD$ as a derived property |
| `0114-derived-prediction-diagnostics.md` | derived prediction diagnostics | accept observed values, predicted values, and provenance; derive all dependent diagnostic arrays once |
| `0115-normalized-path-result-state.md` | normalized path result state | store path-wide policy and split count once; derive predictor-rank-profile selection from candidate arrays |
| `0117-commercial-license-authorship-and-citation.md` | commercial license, authorship, and citation | retain complete BSD-3-Clause terms, name the three copyright holders, and publish software plus companion-paper citation metadata |
| `0118-maintained-figure-labeling-and-axis-policy.md` | maintained figure labeling and axis policy | consistent $\Pi$-PLS notation, factor symbols, numeric dilation ticks, tile titles, path/profile scales, and Tobacco response labels |
| `0119-manuscript-latent-geometry-generator.md` | manuscript latent-geometry generator | additive exact Gaussian manuscript generator with manuscript-oriented immutable truth; existing synthetic and real-data workflows unchanged |
| `0120-companion-manuscript-theory-alignment.md` | companion-manuscript theory alignment | canonical projector/optimization/diagonal derivation, corrected fitted dimension, and explicit manuscript/package scope boundary |
| `0121-canonical-pipls-terminology.md` | canonical Pi-PLS terminology | retained basis/projector, predictor and response directions, dilation, paired modes, score orientation, and public rank-name meanings |
| `0122-public-terminology-propagation.md` | public terminology propagation | canonical paired-mode, rank, direction, dilation, and loading distinctions across living docs and generated docstrings |
| `0123-companion-manuscript-synthetic-data-guide.md` | companion-manuscript synthetic-data guide | distinguish exact distribution, seeded realization, and complete-study reproduction without changing package workflows |
| `0124-mathematical-typography-and-subscripts.md` | mathematical typography and descriptive subscripts | bold complete matrices, upright descriptive subscripts, italic variable indices, and renderable generated equations |
| `0125-retire-benchmark-layer.md` | benchmark-layer retirement | remove benchmark scripts, tests, outputs, navigation, and active contracts; preserve historical records |
| `0126-historical-removal-test-policy.md` | historical-removal test policy | retain negative tests for current boundaries, not one tombstone per pre-release removal |
| `0127-artifact-based-rendering-validation.md` | artifact-based rendering validation | protect rendering ownership, numerical meaning, and generated artifacts without exact Matplotlib source locks |
| `0128-same-file-rendering-functions.md` | same-file rendering functions | keep complete-example analysis in `main()` while private functions in the numbered script own rendering and report writing |
| `0130-mathematical-decomposition-field-names.md` | mathematical decomposition field names | expose `predictor_directions` and `response_directions` without pre-release aliases; retain standard PLS-style estimator rotation attributes |
| `0131-concise-response-standardized-mse-names.md` | concise response-standardized MSE names | expose concise public scorer callables and stable default scorer string without aliases or numerical changes |
| `0132-predicate-search-exhaustiveness-name.md` | predicate-style search exhaustiveness name | expose `search_is_exhaustive_` as the completed-search coverage predicate without an alias or behavioral change |
| `0133-regression-generator-truth-name.md` | regression-generator truth naming | distinguish the configurable regression truth record from the manuscript latent-geometry truth without aliases or data changes |
| `0134-type-revealing-result-properties.md` | type-revealing public result properties | expose explicit result-object, CV-MSE, and boolean predicate names without aliases or numerical changes |
| `0135-specific-predictor-rank-support-warning.md` | specific predictor-rank support warning name | expose `PredictorRankSupportWarning` without an alias or support-policy change |
| `0136-seeded-shuffled-example-folds.md` | seeded shuffled example folds | use reproducible shuffled five-fold partitions in maintained examples while leaving package defaults and leave-one-out unchanged |
| `0137-post-fit-inspect-decide-refit-lifecycle.md` | post-fit inspect-decide-refit lifecycle | make search a path-evidence object; select, refit, and compute OOF diagnostics through explicit post-fit operations |
| `0138-package-owned-pulp-dataset-loader.md` | package-owned Pulp dataset loader | provide one installed immutable `load_pulp()` dataset and archive the former repository layout without a duplicate active copy |
| `0139-three-stage-user-onboarding.md` | three-stage user onboarding | lead with an automatic Pulp fit, then inspect-decide-refit mechanics, then selection-conditioned validation and interpretation |
| `0140-search-owned-path-selection.md` | search-owned path selection | make `PiPLSSearchCV.select()` the sole public selected-row lookup and reduce `PiPLSComponentPath` to aligned numerical evidence |
| `0141-spectral-predictor-rank-profile-figures.md` | spectral predictor-rank profiles | make Sugarcane and Tobacco plot split-SD rank profiles at the fitted model selection, including Tobacco tolerance selection |
| `0142-package-owned-reference-datasets.md` | package-owned reference datasets | extend the named immutable loader and language-neutral package-resource contract from Pulp to Sugarcane and Tobacco without a registry or duplicate active matrices |
| `0143-model-selection-provenance-and-oof-reporting.md` | model-selection provenance and OOF reporting | retain the exact refit selection as `model.selection_` and make `oof_report(selection=...)` reuse every materialized search split |
| `0144-pre-release-public-surface-cleanup.md` | pre-release public-surface cleanup | remove duplicated result access, search aliases, candidate parameter representations, dataset aliases, shape-only properties, and top-level result re-exports while retaining distinct selection, inspection, OOF, and plotting roles |
| `0145-final-implementation-surface-cleanup.md` | final implementation-surface cleanup | remove residual duplicate fitted attributes, privatize model-selection internals, remove unused private helpers, declare remaining module exports, and state fitting-free selection positively |
| `0146-cv-mse-tolerance-selection.md` | CV-MSE tolerance selection and split-SD reporting | replace the 1-SE heuristic with dual-tolerance minimum-CV-MSE selection, descriptive split SD, a 10% Tobacco demonstration, and repeated Pulp validation |
| `0147-decision-lifecycle-and-maintainer-context.md` | decision lifecycle and maintainer-context consolidation | distinguish current decisions, compact historical summaries, and retired records; compact `.llm`, curate the decision set, simplify brittle tests, harden snapshots, and split dataset internals without changing public behavior |

## Current canonical clusters

Use the newest applicable accepted decision when records overlap. The current implementation is
most directly governed by these clusters:

- **Mathematics and numerical construction:** 0001--0009, refined by 0025, 0031--0032, 0039--0040,
  0073, 0086, and 0121--0123.
- **Search, selection, and validation:** 0010--0014, 0039--0040, 0066, 0072--0073, 0102,
  0137, 0140, 0143--0146.
- **Datasets and data ownership:** 0015--0016, 0019, 0022--0025, 0041, 0138, and 0142.
- **Inspection and rendering:** 0042--0045, 0058--0061, 0079--0089, and 0141.
- **Examples and documentation:** 0037--0038, 0044--0057, 0062--0078, 0090--0101, and
  0139.
- **Packaging, compatibility, and repository policy:** 0021, 0024, 0054--0056, 0103--0105,
  0110, 0124--0128, and 0147.
- **Pre-release API normalization:** 0112--0136 and 0144--0146. Later records in this cluster
  supersede intermediate names and migration surfaces.

These ranges are navigation aids, not a substitute for the full records, the retirement map, or
the historical summary that will be added in Decision 0147 Patch 4.

## Implemented clarifications

The implemented package has the following current boundaries regardless of earlier intermediate
records:

- `PiPLSRegression` fits one explicit rank pair; `PiPLSSearchCV` owns path evaluation and post-fit
  selection, refitting, and OOF reporting.
- The only named selection rules are `best_score` and tolerance-based `minimum_cv_mse`.
- CV-MSE dispersion is population SD across materialized splits; no standard-error result or rule
  exists.
- Refitted models retain the exact immutable `model.selection_`; OOF reporting requires an existing
  compatible selection and reuses the fitted search splits.
- Top-level `pipls` exports only the two estimators, `PredictorRankSupportWarning`, and version
  metadata; result records remain public from focused modules.
- Pulp, Sugarcane, and Tobacco are the named package-owned reference datasets, with one active
  language-neutral resource copy each.
- Numerical inspection is package-owned; plotting and report composition are caller-owned.
- The package repository is distinct from downstream paper-reproduction environments.

When adding a decision, create the numbered file and add one table row in the same patch. Retirement
requires the explicit mapping, reference cleanup, and validation defined by Decision 0147.
