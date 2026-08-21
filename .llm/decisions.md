# Current decision registry

This registry lists only numbered decisions that still define current behavior or policy. Use
`docs/decisions/history.md` for compact development history and
`docs/decisions/retirements.md` for the exhaustive retired-filename map.

| Decision | Current subject | Current contract or outcome |
|---|---|---|
| `0001-core-definition.md` | fixed Pi-PLS construction | SVD/least-squares core with explicit `(h, r_pi)` admissibility |
| `0002-preprocessing-semantics.md` | centering and scaling | preprocessing remains outside the fixed numerical core |
| `0003-predictor-rank-selection.md` | historical rank-bound design and conditional selection | materialized split reuse and deterministic low-rank ties remain; Decision 0154 supersedes the general $n/c$ ceiling |
| `0004-response-standardized-mse.md` | selection loss | fold-local response scales and uniform response weighting |
| `0007-predictor-rank-search-policies.md` | exhaustive versus adaptive search | deterministic adaptive and exhaustive algorithms with public `"adaptive"` and `"exhaustive"` values |
| `0008-predictor-svd-policy.md` | scalable predictor decomposition | independent `full`, `randomized`, and `auto` solver policy |
| `0009-public-parameter-validation.md` | exposed controls | early validation and low-statistical-support warning |
| `0014-validation-metadata-scope.md` | groups and weighting boundary | groups-only splitter metadata; no weighted fitting or general routing |
| `0015-dataset-and-synthetic-api.md` | dataset and synthetic boundary | optional immutable datasets plus local seeded latent-structure generation |
| `0024-package-product-repository-boundary.md` | package versus publication ownership | `pipls` owns the software product; paper reproduction and separate block-scaling products stay downstream |
| `0025-model-internal-standardization-boundary.md` | current versus external scaling | estimator centering/scaling is current and fold-local; external learned scaling must remain inside the same CV boundary |
| `0032-full-sample-rank-support.md` | EPV sample-count convention | full supplied $n$ defines the explicit EPV heuristic; centered training folds impose feasibility caps |
| `0039-fixed-estimator-path-search-boundary.md` | estimator versus selection ownership | implemented split: fixed `PiPLSRegression`, triangular selection in `PiPLSSearchCV` |
| `0041-legacy-dataset-licensing-roadmap.md` | legacy dataset licensing and roadmap | retain the three licensed datasets; exclude Corn, legacy Steel, SARCOS, and FRED-MD |
| `0042-model-inspection-and-post-analysis.md` | fitted-model analysis architecture | separate selection diagnostics, interpretation, and prediction diagnostics; reusable inspection, plotting, and all three real-data integrations |
| `0045-pls-family-analysis-boundary.md` | comparison versus fitted-model analysis ownership | retain ordinary PLS for CV-MSE comparison; keep $P$, $D$, and $Q$ Pi-PLS-specific; make shared analysis estimator-neutral and apply it only to Pi-PLS in numbered examples |
| `0054-compatibility-policy.md` | supported interpreter and dependency ranges | Python 3.10–3.14, guarded runtime dependency majors, explicit CI responsibilities, and clean wheel/sdist validation |
| `0061-example-owned-report-composition.md` | complete example rendering ownership | examples create every figure and axis directly from immutable numerical results and own all report composition |
| `0065-documentation-layer-consolidation.md` | tutorial, guide, and reference ownership | self-contained strict docs separate worked tutorials, task guides, scientific interpretation, and generated API reference |
| `0066-immutable-component-path-api.md` | concise path-result API | frozen aligned path arrays and separate immutable search-owned scalar selections without path lookup methods |
| `0072-conditional-predictor-rank-profile.md` | conditional predictor-rank inspection | derive one immutable sorted rank profile on demand from `cv_results_` without another fitted representation |
| `0083-data-first-rendering-policy.md` | final data-first rendering policy | immutable results are the compatibility surface; rendering and local plot helpers remain caller-owned |
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
| `0137-post-fit-inspect-decide-refit-lifecycle.md` | post-fit inspect-decide-refit lifecycle | make search a path-evidence object; select, refit, and compute OOF diagnostics through explicit post-search operations |
| `0140-search-owned-path-selection.md` | search-owned path selection | make `PiPLSSearchCV.select()` the sole public selected-row lookup and reduce `PiPLSComponentPath` to aligned numerical evidence |
| `0141-spectral-predictor-rank-profile-figures.md` | spectral predictor-rank profiles | make Sugarcane and Tobacco plot split-SD rank profiles at the exact selection used for final fitting, including Tobacco tolerance selection |
| `0142-package-owned-reference-datasets.md` | package-owned reference datasets | extend the named immutable loader and language-neutral package-resource contract from Pulp to Sugarcane and Tobacco without a registry or duplicate active matrices |
| `0143-model-selection-provenance-and-oof-reporting.md` | model-selection provenance and OOF reporting | retain exact selection provenance, share compatible selections across OOF reporting and refitting, and keep reports protocol-neutral |
| `0146-cv-mse-tolerance-selection.md` | CV-MSE tolerance selection and split-SD reporting | replace the 1-SE heuristic with dual-tolerance minimum-CV-MSE selection, descriptive split SD, a 10% Tobacco demonstration, and repeated Pulp validation |
| `0147-decision-lifecycle-and-maintainer-context.md` | decision lifecycle and maintainer-context consolidation | distinguish current decisions, historical summaries, and retired records; keep tests behavior-focused and active maintainer context current |
| `0148-predictor-rank-tolerance-selection.md` | predictor-rank tolerance selection | constructor tolerances, immutable rank evidence, conditioned component rules, and separate 10% Tobacco predictor-rank and component-count demonstration |
| `0152-selection-review-feedback-workflow.md` | selection-review feedback workflow | inspect the unselected path before selecting, review conditional evidence with one feedback edge, and reserve qualification or validation for independent assessment |
| `0153-independent-block-scaling-controls.md` | independent predictor and response scaling controls | retain `scale` as the compatibility default while allowing fold-local pipeline predictor scaling and Pi-PLS response scaling to be controlled independently |
| `0154-full-domain-predictor-rank-selection.md` | full-domain predictor-rank selection and explicit EPV policy | implemented exhaustive full-feasible automatic coverage, explicit `"epv"`, and removal of the pre-release `"max"`/`"rule"` rank shortcuts |
| `0155-response-subspace-selection-policies.md` | response-subspace selection policies | closed implementation: cross-covariance default plus an explicit least-squares/RRR-inspired software extension outside the peer-reviewed publication |
| `0156-unified-pls-family-path-comparison.md` | unified PLS-family path comparison | consolidate both Pi-PLS response policies and ordinary PLS into Example 03 on shared materialized folds for all three reference datasets |
| `0157-near-saturated-synthetic-pls-comparison.md` | near-saturated synthetic PLS-family stress case | closed exploratory extension of Example 03 with a fixed deterministic 25-by-40, 10-response design, matched folds, and exhaustive Pi-PLS rank coverage |
| `0158-home-page-parsimony-comparison.md` | Home-page parsimony comparison | closed documentation contract: simplified Pulp/Tobacco Home figures share the Example-03 protocol, show publication-default Pi-PLS versus PLS, and bound the parsimony claim to shared component count |
| `0159-documentation-cross-reference-architecture.md` | documentation cross-reference architecture | active four-patch navigation contract: canonical anchors plus semantic dataset, companion-publication, and concept-specific theory links are implemented; final navigation audit and regression protection remain |

## Implemented clarifications

- `PiPLSRegression` fits one explicit pair; `PiPLSSearchCV` owns path evaluation, selection, OOF
  reporting, and explicit final refitting.
- Predictor-rank search implements exhaustive full-feasible automatic coverage by default, with
  `"adaptive"` as an explicit reduced-coverage option and `"epv"` as the fixed-rank
  events-per-variable-inspired policy under Decision 0154.
- The only named component-count rules are `best_score` and tolerance-based `minimum_cv_mse`;
  `cv_mse_std` is descriptive population split SD.
- OOF reporting and final refitting may consume the same compatible immutable selection. OOF reports
  are selection-conditioned and protocol-neutral; there is no dedicated leave-one-out surface.
- Pulp, Sugarcane, and Tobacco are the closed package-owned reference-dataset set.
- Numerical inspection is package-owned; plotting, local rendering helpers, and report composition
  are caller-owned.
- Tests protect behavior and machine-readable outputs. Complete documentation, examples, and
  installed artifacts are validated by their dedicated Make targets.

When adding a decision, create the numbered file and index it in the same patch. Retirement requires
an explicit mapping, active-reference cleanup, and the validation defined by Decision 0147.
