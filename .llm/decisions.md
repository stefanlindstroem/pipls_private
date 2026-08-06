# Current decision registry

This registry lists only numbered decisions that still define current behavior or policy. Use
`docs/decisions/history.md` for compact development history and
`docs/decisions/retirements.md` for the exhaustive retired-filename map.

| Decision | Current subject | Current contract or outcome |
|---|---|---|
| `0001-core-definition.md` | fixed Pi-PLS construction | SVD/least-squares core with explicit `(h, r_pi)` admissibility |
| `0002-preprocessing-semantics.md` | centering and scaling | preprocessing remains outside the fixed numerical core |
| `0003-predictor-rank-selection.md` | rank bound and conditional selection | ceiling rule, materialized splits, deterministic low-rank ties; sample-count convention refined by 0032 |
| `0004-response-standardized-mse.md` | selection loss | fold-local response scales and uniform response weighting |
| `0007-predictor-rank-search-policies.md` | exhaustive versus adaptive search | deterministic adaptive and exhaustive algorithms; public value migration is governed by 0149 |
| `0008-predictor-svd-policy.md` | scalable predictor decomposition | independent `full`, `randomized`, and `auto` solver policy |
| `0009-public-parameter-validation.md` | exposed controls | early validation and low-statistical-support warning |
| `0014-validation-metadata-scope.md` | groups and weighting boundary | groups-only splitter metadata; no weighted fitting or general routing |
| `0015-dataset-and-synthetic-api.md` | dataset and synthetic boundary | optional immutable datasets plus local seeded latent-structure generation |
| `0024-package-product-repository-boundary.md` | package versus publication ownership | `pipls` owns the software product; paper reproduction stays downstream; future block-aware API design is deferred |
| `0025-model-internal-standardization-boundary.md` | current versus future scaling | estimator centering/scaling is current and fold-local; only future block-aware variants are deferred |
| `0032-full-sample-rank-support.md` | rank-support sample-count convention | full supplied $n$ defines support; centered training folds impose feasibility caps |
| `0039-fixed-estimator-path-search-boundary.md` | estimator versus selection ownership | implemented split: fixed `PiPLSRegression`, triangular selection in `PiPLSSearchCV` |
| `0041-legacy-dataset-licensing-roadmap.md` | legacy dataset licensing and roadmap | retain the three licensed datasets; exclude Corn, legacy Steel, SARCOS, and FRED-MD |
| `0042-model-inspection-and-post-analysis.md` | fitted-model analysis architecture | separate selection diagnostics, interpretation, and prediction diagnostics; reusable inspection, plotting, and all three real-data integrations |
| `0045-pls-family-analysis-boundary.md` | comparison versus fitted-model analysis ownership | retain ordinary PLS for CV-MSE comparison; keep $P$, $D$, and $Q$ Pi-PLS-specific; make shared analysis estimator-neutral and apply it only to Pi-PLS in numbered examples |
| `0054-compatibility-policy.md` | supported interpreter and dependency ranges | Python 3.10–3.14, guarded runtime dependency majors, explicit CI responsibilities, and clean wheel/sdist validation |
| `0061-example-owned-report-composition.md` | complete example rendering ownership | examples create every figure and axis directly from immutable numerical results and own all report composition |
| `0065-documentation-layer-consolidation.md` | tutorial, guide, and reference ownership | self-contained strict docs separate worked tutorials, task guides, scientific interpretation, and generated API reference |
| `0066-immutable-component-path-api.md` | concise path-result API | frozen aligned path arrays and separate immutable search-owned scalar selections without path lookup methods |
| `0072-conditional-predictor-rank-profile.md` | conditional predictor-rank inspection | derive one immutable sorted rank profile on demand from `cv_results_` without another fitted representation |
| `0083-data-first-rendering-policy.md` | final data-first rendering policy | immutable results are the compatibility surface; rendering remains optional and caller-owned |
| `0091-clean-git-snapshots.md` | clean committed-tree snapshots | refuse tracked, staged, or nonignored untracked changes and archive `HEAD` so ignored local files cannot enter handoffs |
| `0092-fold-numerical-rank-feasibility.md` | fold numerical-rank feasibility | cap path candidates by the minimum rank verified after fold-local preprocessing before scoring |
| `0093-public-result-invariants.md` | immutable core public-result invariants | validate direct construction, defensive copies, scalar normalization, OOF coverage, and pickle reconstruction |
| `0094-inspection-result-safety.md` | immutable and finite inspection results | validate direct construction and pickle reconstruction; use range-safe calculations and reject unrepresentable derived values |
| `0102-path-search-defaults.md` | path-search defaults and scorer presentation | default to selection-only `refit=False`; use a stable package scorer name resolving to the public callable |
| `0103-installation-and-optional-dependency-boundary.md` | installation and optional dependencies | retain only maintained `dev`, `examples`, and `docs` extras; use noneditable public installation and editable contributor setup |
| `0110-response-anchored-display-factors.md` | response-anchored Pi-PLS display factors | retain predictor-canonical defaults; optionally orient every component by a selected response row and requested sign |
| `0117-commercial-license-authorship-and-citation.md` | commercial license, authorship, and citation | retain complete BSD-3-Clause terms, name the three copyright holders, and publish software plus companion-paper citation metadata |
| `0119-manuscript-latent-geometry-generator.md` | manuscript latent-geometry generator | additive exact Gaussian manuscript generator with manuscript-oriented immutable truth; existing synthetic and real-data workflows unchanged |
| `0120-companion-manuscript-theory-alignment.md` | companion-manuscript theory alignment | canonical projector/optimization/diagonal derivation, corrected fitted dimension, and explicit manuscript/package scope boundary |
| `0121-canonical-pipls-terminology.md` | canonical Pi-PLS terminology | retained basis/projector, predictor and response directions, dilation, paired modes, score orientation, and public rank-name meanings |
| `0123-companion-manuscript-synthetic-data-guide.md` | companion-manuscript synthetic-data guide | distinguish exact distribution, seeded realization, and complete-study reproduction without changing package workflows |
| `0124-mathematical-typography-and-subscripts.md` | mathematical typography and descriptive subscripts | bold complete matrices, upright descriptive subscripts, italic variable indices, and renderable generated equations |
| `0127-artifact-based-rendering-validation.md` | artifact-based rendering validation | protect rendering ownership, numerical meaning, and generated artifacts without exact Matplotlib source locks |
| `0137-post-fit-inspect-decide-refit-lifecycle.md` | post-fit inspect-decide-refit lifecycle | make search a path-evidence object; select, refit, and compute OOF diagnostics through explicit post-search operations |
| `0139-three-stage-user-onboarding.md` | three-stage user onboarding | lead with an automatic Pulp fit, then inspect-decide-refit mechanics, then selection-conditioned validation and interpretation |
| `0140-search-owned-path-selection.md` | search-owned path selection | make `PiPLSSearchCV.select()` the sole public selected-row lookup and reduce `PiPLSComponentPath` to aligned numerical evidence |
| `0141-spectral-predictor-rank-profile-figures.md` | spectral predictor-rank profiles | make Sugarcane and Tobacco plot split-SD rank profiles at the exact selection used for final fitting, including Tobacco tolerance selection |
| `0142-package-owned-reference-datasets.md` | package-owned reference datasets | extend the named immutable loader and language-neutral package-resource contract from Pulp to Sugarcane and Tobacco without a registry or duplicate active matrices |
| `0143-model-selection-provenance-and-oof-reporting.md` | model-selection provenance and OOF reporting | retain the exact refit selection as `model.selection_` and make `oof_report(selection=...)` reuse every materialized search split |
| `0145-final-implementation-surface-cleanup.md` | final implementation-surface cleanup | remove residual duplicate fitted attributes, privatize model-selection internals, remove unused private helpers, declare remaining module exports, and state fitting-free selection positively |
| `0146-cv-mse-tolerance-selection.md` | CV-MSE tolerance selection and split-SD reporting | replace the 1-SE heuristic with dual-tolerance minimum-CV-MSE selection, descriptive split SD, a 10% Tobacco demonstration, and repeated Pulp validation |
| `0147-decision-lifecycle-and-maintainer-context.md` | decision lifecycle and maintainer-context consolidation | distinguish current decisions, compact historical summaries, and retired records; normalize `.llm`, decision links, structural tests, snapshot hygiene, and private dataset ownership without changing public behavior |
| `0148-predictor-rank-tolerance-selection.md` | predictor-rank tolerance selection | constructor tolerances, immutable rank evidence, conditioned component rules, and separate 10% Tobacco predictor-rank and component-count demonstration |
| `0149-predictor-rank-search-terminology.md` | predictor-rank search terminology | use `"adaptive"`/`"exhaustive"`, preserve algorithms and achieved-coverage diagnostics, reject retired values, and reject inapplicable nondefault exhaustive mode |
| `0150-computational-performance-guidance.md` | computational-performance guidance | central reference page, explicit cost categories, fold-local preprocessing, reproducible examples, and current implementation semantics |
| `0151-selection-driven-refit-workflow.md` | selection-driven refit workflow | pass one compatible immutable selection through OOF reporting and final refitting; reorder analytical examples and add source-level tutorial flowcharts |
| `0152-selection-review-feedback-workflow.md` | selection-review feedback workflow | inspect the unselected path before selecting, review conditional evidence with one feedback edge, and reserve qualification or validation for independent assessment |
| `0153-remove-package-owned-leave-one-out-support.md` | remove package-owned leave-one-out support | retain generic splitter interoperability and OOF reporting while removing the detector, provenance field, dedicated example, support claims, and tests in five patches |

## Current canonical clusters

- **Mathematics and numerical construction:** 0001--0004, 0007--0009, 0014, 0025, 0032, 0092,
  0120--0121, and 0146--0149.
- **Estimator, search, and result ownership:** 0039, 0066, 0093, 0102, 0137, 0140, 0143,
  0145--0149, and 0151--0153.
- **Datasets and product scope:** 0015, 0024--0025, 0041, 0119, 0123, and 0142.
- **Inspection and rendering:** 0042, 0045, 0061, 0083, 0094, 0110, 0124, 0127, and 0141.
- **Documentation, compatibility, and repository policy:** 0054, 0065, 0091, 0103, 0117, 0139,
  0147, 0150, and 0151--0153.

## Accepted Decision 0153 contract

- Remove package-owned leave-one-out detection, provenance, examples, documentation, and tests.
- Retain generic scikit-learn-compatible splitters, explicit split iterables, and protocol-neutral
  OOF reporting.
- Keep singleton-validation scorer safety without leave-one-out-specific wording or imports.
- Do not add a compatibility alias, deprecation scaffold, replacement helper, or dedicated mode.
- Implement the removal in five patches; Patch 1 changes decisions and maintainer contracts only.

## Implemented Decision 0152 contract

- Manual component-count tutorials inspect the unselected path before creating a selection.
- Setting the chosen count and calling `search.select(...)` are one conceptual operation.
- Selected-path, conditional-rank, and same-search OOF results form one review stage with a
  possible feedback edge to selection; they are not independent qualification.
- Tutorial 2 and Tutorial 3 expose separate unselected and selected component-path artifacts.
- The accepted selection remains the exact object passed to final refitting.

## Implemented Decision 0151 contract

- Add `selection=` to `PiPLSSearchCV.refit()` while retaining rule-based and component-count routes.
- Use one exact compatibility validator for selections consumed by `refit()` and `oof_report()`.
- Reorder evidence-retaining examples around search evidence, one selection, optional OOF
  inspection, and final refitting from that same selection.
- Keep validation-only and comparison-only routes free of unnecessary final models.
- Add one supplementary vertical Mermaid flowchart plus equivalent prose to each served tutorial,
  without committed generated diagram assets.
- The six-patch implementation is complete; Decision 0152 refines the manual tutorial review
  sequence without changing the runtime handoff.

## Implemented Decision 0150 contract

- Add one canonical served guide at `docs/computational_performance.md`, positioned after
  path-selection details and before troubleshooting.
- Separate validation evidence, candidate policy, numerical approximation, parallel execution, and
  repeated diagnostic work.
- Preserve fold-local preprocessing and use explicit seeds for shuffled CV and randomized SVD.
- Document current candidate, parallelism, OOF, and timing behavior without universal benchmark
  claims.
- Public guides route to the central page; semantic documentation tests and isolated
  source-distribution builds protect its executable examples, shipped source, and rendered page.

## Implemented Decision 0149 contract

- Keep the `search_method` parameter and rename its values to `"adaptive"` (default) and
  `"exhaustive"` without compatibility aliases.
- Preserve the existing candidate-generation algorithms and `search_is_exhaustive_` as the
  achieved-coverage diagnostic.
- Omit `search_method` from fixed and maximum-rank examples, accept the default for constructor
  consistency, and reject the inapplicable nondefault exhaustive method.
- The terminology migration is complete; Decision 0150 may now add computational-performance
  guidance against the final values.

## Implemented Decision 0148 contract

- Separate constructor-level relative and absolute tolerances govern conditional predictor-rank
  retention; component-count tolerances remain post-search controls.
- Adaptive refinement and `rank_test_score` use private exact-score comparison; public
  predictor-rank tolerances act only after candidates have been evaluated.
- Named component-count rules operate on the rank-conditioned path, and optimized rows expose
  immutable exact-reference and tolerance provenance.

## Implemented clarifications

- `PiPLSRegression` fits one explicit pair; `PiPLSSearchCV` owns path evaluation and post-search
  selection, refitting, and OOF reporting.
- The only named selection rules are `best_score` and tolerance-based `minimum_cv_mse`.
- CV-MSE dispersion is population SD across materialized splits; no standard-error result or rule
  exists.
- Refitted models retain the exact immutable `model.selection_`; under Decision 0151, OOF
  reporting and final refitting can consume the same pre-existing compatible selection while reusing
  the fitted search splits and averaging repeated validation predictions per observation.
- Pulp, Sugarcane, and Tobacco are the closed set of package-owned reference datasets. Arbitrary
  user data remains ordinary array-like `X` and `y`.
- Numerical inspection is package-owned; plotting and report composition are caller-owned.
- The repository is the software product. Paper-reproduction environments remain downstream.

When adding a decision, create the numbered file and index it in the same patch. Retirement requires
an explicit mapping, active-reference cleanup, and the validation defined by Decision 0147.
