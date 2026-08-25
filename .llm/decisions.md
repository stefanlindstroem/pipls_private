# Current decision registry

This registry lists only numbered decisions that still define current behavior or policy. Use
`docs/decisions/history.md` for compact development history and
`docs/decisions/retirements.md` for the exhaustive retired-filename map.

| Decision | Current subject | Current contract or outcome |
|---|---|---|
| `0001-core-definition.md` | fixed Pi-PLS construction | SVD/least-squares core with explicit `(h, r_pi)` admissibility |
| `0002-preprocessing-semantics.md` | centering and scaling | preprocessing remains outside the fixed numerical core |
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
| `0066-immutable-component-path-api.md` | concise path-result API | frozen aligned path arrays and separate immutable search-owned scalar selections without path lookup methods |
| `0072-conditional-predictor-rank-profile.md` | conditional predictor-rank inspection | derive one immutable sorted rank profile on demand from `cv_results_` without another fitted representation |
| `0083-data-first-rendering-policy.md` | final data-first rendering policy | immutable results are the compatibility surface; rendering and local plot helpers remain caller-owned |
| `0091-clean-git-snapshots.md` | clean committed-tree snapshots | refuse tracked, staged, or nonignored untracked changes and archive `HEAD` so ignored local files cannot enter handoffs |
| `0092-fold-numerical-rank-feasibility.md` | fold numerical-rank feasibility | cap path candidates by the minimum rank verified after fold-local preprocessing before scoring |
| `0093-public-result-invariants.md` | immutable core public-result invariants | validate direct construction, defensive copies, scalar normalization, OOF coverage, and pickle reconstruction |
| `0094-inspection-result-safety.md` | immutable and finite inspection results | validate direct construction and pickle reconstruction; use range-safe calculations and reject unrepresentable derived values |
| `0103-installation-and-optional-dependency-boundary.md` | installation and optional dependencies | retain only maintained `dev`, `examples`, and `docs` extras; use noneditable public installation and editable contributor setup |
| `0110-response-anchored-display-factors.md` | response-anchored Pi-PLS display factors | retain predictor-canonical defaults; optionally orient every component by a selected response row and requested sign |
| `0117-commercial-license-authorship-and-citation.md` | commercial license, authorship, and citation | retain complete BSD-3-Clause terms, name the three copyright holders, and publish software plus companion-paper citation metadata |
| `0119-manuscript-latent-geometry-generator.md` | manuscript latent-geometry generator | additive exact Gaussian manuscript generator with manuscript-oriented immutable truth; existing synthetic and real-data workflows unchanged |
| `0120-companion-manuscript-theory-alignment.md` | companion-manuscript theory alignment | canonical projector/optimization/diagonal derivation, corrected fitted dimension, and explicit manuscript/package scope boundary |
| `0121-canonical-pipls-terminology.md` | canonical Pi-PLS terminology | retained basis/projector, predictor and response directions, dilation, paired modes, score orientation, and public rank-name meanings |
| `0124-mathematical-typography-and-subscripts.md` | mathematical typography and descriptive subscripts | bold complete matrices, upright descriptive subscripts, italic variable indices, and renderable generated equations |
| `0142-package-owned-reference-datasets.md` | package-owned reference datasets | extend the named immutable loader and language-neutral package-resource contract from Pulp to Sugarcane and Tobacco without a registry or duplicate active matrices |
| `0143-model-selection-provenance-and-oof-reporting.md` | model-selection provenance and OOF reporting | retain exact selection provenance, share compatible selections across OOF reporting and refitting, keep reports protocol-neutral, and treat same-search OOF diagnostics as selection-conditioned inspection rather than independent validation |
| `0146-cv-mse-tolerance-selection.md` | CV-MSE tolerance selection and split-SD reporting | replace the 1-SE heuristic with dual-tolerance minimum-CV-MSE selection, descriptive split SD, a 10% Tobacco demonstration, and repeated Pulp validation |
| `0147-decision-lifecycle-and-maintainer-context.md` | decision lifecycle and maintainer-context consolidation | distinguish current decisions, historical summaries, and retired records; keep tests behavior-focused and active maintainer context current |
| `0148-predictor-rank-tolerance-selection.md` | predictor-rank tolerance selection | constructor tolerances, immutable rank evidence, conditioned component rules, and separate 10% Tobacco predictor-rank and component-count demonstration |
| `0153-independent-block-scaling-controls.md` | independent predictor and response scaling controls | retain `scale` as the compatibility default while allowing fold-local pipeline predictor scaling and Pi-PLS response scaling to be controlled independently |
| `0154-full-domain-predictor-rank-selection.md` | full-domain predictor-rank selection and explicit EPV policy | implemented exhaustive full-feasible automatic coverage, explicit `"epv"`, and removal of the pre-release `"max"`/`"rule"` rank shortcuts |
| `0155-response-subspace-selection-policies.md` | response-subspace selection policies | closed implementation: cross-covariance default plus an explicit least-squares/RRR-inspired software extension outside the peer-reviewed publication |
| `0164-lean-reference-architecture.md` | lean reference architecture | lean flat lookup reference; tutorials own workflows, domain pages own exact contracts, and synthetic generator explanation plus the maintained latent-role figure remain served |
| `0165-selection-evidence-and-oof-diagnostic-boundary.md` | selection evidence and OOF diagnostic boundary | component path and optional conditional rank evidence complete the documented selection; same-search OOF reporting diagnoses that accepted selection and remains selection-conditioned |

## Implemented clarifications

- `PiPLSRegression` fits one explicit pair; `PiPLSSearchCV` owns path evaluation, selection, OOF
  reporting, and explicit final refitting.
- Predictor-rank search implements exhaustive full-feasible automatic coverage by default, with
  `"adaptive"` as an explicit reduced-coverage option and `"epv"` as the fixed-rank
  events-per-variable-inspired policy under Decision 0154.
- The only named component-count rules are `best_score` and tolerance-based `minimum_cv_mse`;
  `cv_mse_std` is descriptive population split SD.
- OOF reporting and final refitting may consume the same compatible immutable selection. Under
  Decision 0165, maintained workflows complete ordinary selection from path and optional conditional
  rank evidence before OOF inspection. OOF reports are selection-conditioned and protocol-neutral;
  there is no dedicated leave-one-out surface.
- Pulp, Sugarcane, and Tobacco are the closed package-owned reference-dataset set.
- Numerical inspection is package-owned; plotting, local rendering helpers, and report composition
  are caller-owned. Maintained Pulp biplots may use optional line-aware `textalloc` placement but
  must fall back to ordinary Matplotlib endpoint labels when it is unavailable.
- Example 03 is the sole maintained PLS-family path comparison and evaluates both Pi-PLS response
  policies and ordinary PLS on the same materialized folds. Comparative paths are model-development
  evidence rather than independent post-selection validation.
- Complete real-data OOF scalar figures prefer response-wise selection-conditioned OOF $R^2$;
  standardized RMSE remains a public numerical diagnostic, and OOF versus fitted-value provenance
  remains explicit.
- Documentation cross-references are semantic and contextual. Strict documentation builds own link
  resolution; pytest does not pin prose-level link placement.
- Tests protect behavior and machine-readable outputs. Complete numbered examples run through
  `make examples` in dedicated Python-3.12 CI; complete tutorial rendering and documentation run
  through `make docs`; installed artifacts retain their dedicated validation targets.
- Decision numbers are not reused for new records. The inherited 0153/0154 retirement-map
  collisions are frozen exceptions and `make decision-check` rejects any additional reuse.

When adding a decision, create the numbered file and index it in the same patch. Retirement requires
an explicit mapping, active-reference cleanup, `make decision-check`, and the validation defined by
Decision 0147.
