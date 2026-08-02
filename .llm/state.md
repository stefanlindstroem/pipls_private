# Current development state

## Purpose

This is the concise handoff document for starting work from a repository snapshot without prior
chat history. It records the current implemented boundary, accepted owner decisions, deferred
scope, and next admissible increment. Update it whenever a patch changes a phase, public default,
supported composition boundary, product scope, or roadmap order.

A new chat should read this file and `.llm/product_scope.md` before proposing implementation work.
Do not infer current state from an earlier conversation, an old patch, or superseded planning
material alone.

## Implemented boundary

Numbered examples are self-contained user tasks rather than publication-oriented or
context-free API demonstrations. The former context-free advanced-validation script remains
removed; example 02 supplies the short synthetic path-selection tutorial and independent-test
prediction workflow, while example 03 gives leave-one-out validation one focused small-calibration
use case.

Phases A through F4 are complete and committed. The former E4 benchmark sequence was completed
and later retired by Decision 0125 after its development-validation purpose had been served:

- repository, packaging, clean committed-tree root-relative snapshots, and direct Git patch
  workflow;
- example output directories represented by committed `.gitkeep` files in both Git and source
  distributions, with generated PDFs excluded from snapshots and example 01 exercised from an
  extracted source distribution;
- fixed-parameter Pi-PLS numerical core;
- scikit-learn-compatible fixed-model `PiPLSRegression` for one explicit
  `(n_components, predictor_rank)` pair;
- independent full, randomized, and automatic predictor-SVD policies;
- hardened public validation and `PredictorRankSupportWarning` for direct fixed fits with
  fewer than three observations per retained predictor-rank direction;
- transactional fixed and path fits, range-safe boundary preprocessing, safe read-only or
  overlapping `copy=False` inputs, and finite public fitted/output values;
- pipeline-aware `PiPLSSearchCV` for triangular `(n_components, predictor_rank)` search;
- the direct pre-release `PiPLSSearchCV` public name and `src/pipls/search.py` implementation,
  with no former-name alias and unchanged component-path result terminology;
- immutable `PiPLSComponentPath` row arrays with path-wide policy and split-count scalars, derived
  fold-based CV-MSE standard errors, and frozen scalar lookup through `component_path_`;
- explicit post-fit full-data refitting and selection-conditioned OOF reporting from a named rule
  or manually chosen component-path row, with exact split reuse and global `best_*` evidence kept
  separate from each post-fit choice;
- on-demand immutable `PiPLSPredictorRankProfile` results through
  `predictor_rank_profile(n_components)`, derived from `cv_results_`;
- a literal-matrix first example showing one fixed fit, prediction, and decomposition plot without CV;
- a public documentation entry that defines paired latent variables, component-count scanning,
  and elbow-based CV-MSE interpretation before specialized terminology;
- shared private fold-evaluation and adaptive-search machinery;
- PLS-style fitted attributes, feature names, pandas output, inverse reconstruction, and a public
  immutable `PiPLSDecomposition` limited to interpretable directions, dilation, rank diagnostics,
  solver provenance, and the derived centered/scaled regression map;
- grouped, repeated, predefined, temporal, and leave-one-out split workflows, with a focused
  small-sample LOO example reporting ordered OOF predictions and pooled OOF $R^2$;
- optional ordered OOF predictions and immutable `PiPLSValidationReport` composed from the
  selected `PiPLSComponentResult`, with explicit fixed-parameter versus selection-conditioned
  labeling;
- immutable validated `PiPLSDataset` with recursively frozen metadata, explicit rejection of
  object-dtype metadata arrays, and deterministic synthetic generators with shared,
  predictor-specific, and response-specific latent structure;
- a transparent real-data input contract: users and examples read `X` and `Y` explicitly, with no
  metadata, registry, or package-owned loader required for fitting;
- a repository real-dataset convention using comma-delimited `X.csv`, `Y.csv`, and documentary
  `metadata.yaml`;
- a current reference suite containing pulp, sugarcane, and tobacco;
- a completed licensing review that intentionally excludes Corn, the legacy Citrination Steel
  table, SARCOS, and FRED-MD from repository redistribution;
- pure immutable Pi-PLS display factors and standardized explicit-provenance prediction
  diagnostics under `pipls.inspection`;
- estimator-neutral latent-structure, biplot, observation-diagnostic, and prediction-diagnostic
  names that accept compatible fitted Pi-PLS and ordinary PLS models;
- a data-first biplot boundary: `biplot_coordinates()` owns balanced numerical coordinates, while
  the Pulp example and tutorial use direct Matplotlib arrows and optional `adjustText` label layout;
- direct standard PLS-family inspection rendering: scores, loadings, coefficients, and raw
  observation diagnostics are plotted from immutable arrays with ordinary Matplotlib;
- direct Pi-PLS factor rendering from immutable `PiPLSDisplayFactors` arrays with ordinary
  Matplotlib; the package exposes no plotting submodule or convenience renderer;
- direct Pulp, Sugarcane, and Tobacco reference workflows that keep `component_path_`, fixed-model
  OOF predictions, and immutable inspection results in memory and write explicit final PDF figures;
  Sugarcane and Tobacco separate `main()`-owned analysis from private same-file rendering;
- a two-tier tutorial route: a short deterministic synthetic selection-and-prediction workflow
  from example 02, followed by a focused Pulp real-data workflow from example 05; both use
  checked snippets and deterministic single-chart SVG assets;
- direct Tobacco response pagination through two same-file multipage report writers, with full
  predictor SVD, decreasing-wavenumber spectral axes, and raw observation diagnostics;
- a direct Pi-PLS/ordinary-PLS comparison whose two immutable component paths are plotted together
  in example 04 without DataFrame conversion, generated CSV intermediates, or a plotting helper;
- a tutorial-first served site whose synthetic entry tutorial owns the minimum selection contract
  and whose second Pulp tutorial owns real-data selection qualification, selection-conditioned
  OOF analysis, and representative interpretation plots including both $P$ and $QD$ factor views;
  both link to the generated API and advanced references;
- a compact documentation navigation without separate quickstart, estimator, parameter-selection,
  or preprocessing guides: the home page provides the minimal fixed fit, generated API pages own
  exact estimator contracts, and retained path and validation pages cover advanced behavior;
- an audience-oriented documentation entry: the root README owns restrained application-oriented
  motivation, installation, two compact workflows, and tutorial routes; `CONTRIBUTING.md` owns
  development and repository maintenance; served navigation separates programming reference from
  project validation;
- tutorial openings that present purpose, coverage, setup or data, and modeling workflow before
  source provenance, renderer ownership, and figure-generation commands in terminal reproduction
  sections;
- a completed documentation reference layer with a public result-object map, task-oriented
  troubleshooting, generic local-link and anchor validation, and tests that protect structure
  without freezing explanatory prose.
- a served example catalogue whose dataset-specific selection notes and cross-example output or
  rendering notes occupy separate heading scopes, so a narrow Tobacco subsection does not govern
  generic documentation that follows it.
- a consolidated programming reference with eight navigation entries: small generated result and
  utility groups live with their owning estimators, path selection and cross-validation share one
  advanced page, and model inspection retains all stable interpretation anchors without repeating
  elementary plotting recipes.
- documentation and maintained example constructions aligned with the current public defaults:
  `PiPLSSearchCV()` demonstrates path-evaluation operation, decomposition documentation includes
  rank/solver diagnostics, and Pulp rank profiles use the public lookup method;
- a grouped self-documenting maintainer command index that presents `make install` and `make check`
  first, then separates development, documentation/example, and distribution/maintenance targets
  without renaming or changing any recipe.
- a current-boundary testing policy that retires one-off migration tombstones while preserving
  negative tests for explicit public and architectural exclusions.

Decision 0086 reduces the public decomposition to quantities used for fitted-model interpretation;
private construction matrices remain in `PiPLSCoreResult`. Decision 0087 removes scorer plumbing,
exact rotation aliases, adaptive-search bookkeeping, and flat OOF duplicates from the public fitted
surface. Decision 0088 completes API3 by removing result-record bookkeeping and redundant zero
loading blocks and by presenting returned records without constructor-first generated signatures.

Decisions 0079--0083 establish and enforce data-first rendering for biplots, prediction
diagnostics, standard PLS-family inspection, and Pi-PLS factors. Decision 0084 completes the
prediction-diagnostic figure trio. Decision 0085 makes Tutorial 2 sequential and self-contained:
setup and display choices precede use, and every displayed interpretation figure has a matching
standalone renderer snippet. Decision 0089 balances the tutorial's Pi-PLS-specific section by
displaying both $P$ and $QD$, while the complete example retains the separate $D$ and $Q$ plots.
Decision 0090 consolidates the programming reference without changing public objects or numerical
behavior.
Decision 0091 makes every successful handoff snapshot a clean `HEAD` archive: tracked, staged, and
nonignored untracked changes are refused, while ignored generated files are excluded by
construction. Decision 0092 makes the path ceiling respect the minimum predictor rank verified
after fold-local preprocessing, so rank-deficient folds bound the candidate path instead of
aborting it. The third pre-release hardening increment aligns `rank_test_score` with selection by
using one reference-anchored tolerant comparison and forbidding adjacent near-tie chaining.
Decision 0093 makes the core public result records validate direct construction, defensive copies,
scalar values, aligned arrays, OOF coverage, and pickle reconstruction uniformly. Decision 0094
applies the same defensive boundary to inspection records and requires inspection helpers to return
finite float64 quantities or fail explicitly when a derived value is not representable.
Decision 0095 requires every fixed estimator to state its complete rank pair. Decision 0096
distinguishes mathematical $Y$ from scikit-learn `y`, defines both resolved path ceilings before
policy details, and groups inspection concepts with the generated inspection API. Decision 0097
adds one focused small-sample leave-one-out workflow without restoring a context-free splitter
catalogue.
Decision 0098 retains every maintained Make target while grouping the command index around setup,
routine validation, development, documentation and examples, and distribution maintenance.
Decision 0099 restores the example artifact boundary: only output-directory placeholders are
committed or snapshotted, source distributions include those placeholders, and distribution
validation runs the introductory example from a clean extraction.
Decision 0100 moves rendered documentation into a dedicated GitHub Pages workflow: every push and
pull request validates the strict checkout and source-distribution builds, while `master` pushes
receive repository-derived canonical URLs and deploy the built site. The README routes GitHub users
through the current Pages deployment without hard-coding an unconfirmed owner name. Decision 0101
renumbers maintained examples continuously from 01 to 07. Decision 0102 separated path evaluation
from automatic constructor-time refitting and gave the default scorer a stable package string. Decision 0103
retains only the maintained `dev`, `examples`, and `docs` extras. Decision 0104 refines the new-user
documentation route, and Decision 0105 aligns maintained documentation and examples with the
resulting implementation. Decision 0137 now supersedes Decision 0102's former constructor-refit
surface with explicit post-fit refitting.
Plotting migration G1--G5 and public-result cleanup API1--API3 are
complete. Decision 0042 defines the staged fitted-model architecture, and Decision
0045 corrects the
analysis-model boundary. The shared API and numbered-example migrations are complete. Ordinary PLS
is retained only in the dedicated example-04 CV-MSE comparison. Examples
05–07 evaluate Pi-PLS paths only, and every post-analysis quantity comes from the selected Pi-PLS
model.

The current top-level package exports are:

```python
from pipls import (
    PiPLSComponentPath,
    PiPLSComponentResult,
    PiPLSDecomposition,
    PiPLSSearchCV,
    PiPLSPredictorRankProfile,
    PiPLSRegression,
    PiPLSValidationReport,
    PredictorRankSupportWarning,
)
```

Dataset functionality is public from `pipls.datasets`:

```python
from pipls.datasets import (
    PiPLSDataset,
    PiPLSRegressionTruth,
    make_pipls_regression,
    make_pipls_train_test,
)
```

## Accepted repository-product boundary

Decision 0024 defines `pipls` as a long-lived software-product repository. It owns:

- the installable package and public API;
- user documentation and numbered examples from a minimal fixed fit through complete analyses;
- transparent reference datasets;
- tests, packaging, compatibility policy, and releases.

It does not own manuscript figures, complete publication experiment grids, paper-only OLS/CCA
comparisons, cached paper results, or publication-specific environments. Those belong in downstream
reproduction repositories that pin tagged `pipls` releases.

Ordinary PLS remains the comparator in the dedicated component-path example because it is the nearest practical baseline
for Pi-PLS users. OLS or CCA are included only when they protect a package-level identity, limiting
case, or public behavior.

## Accepted public defaults and boundaries

| Concern | Current contract |
|---|---|
| Conditional predictor-rank selection | `PiPLSSearchCV(search_method="auto")` |
| Component counts | `n_components_values="all"` by default; explicit integer sequences request a subset |
| Path search | `PiPLSSearchCV(search_method="auto")` by default |
| Component-path artifact | immutable `component_path_` with aligned score, CV-MSE, fold-SD, derived fold-based standard-error, predictor-rank, policy, and split-count arrays plus scalar lookup |
| Conditional rank profile | `predictor_rank_profile(h)` returns evaluated ranks and aligned score/CV-MSE arrays on demand, plus the selected scalar row |
| Exhaustive search | explicit `PiPLSSearchCV(search_method="optimal")` |
| Predictor SVD | `svd_solver="auto"`, with the documented conservative threshold |
| Reproducibility | estimator `random_state` accepts integer, NumPy `RandomState`, or `None`; default `0` is reproducible |
| Rank support rule | path-only `samples_per_predictor_rank=5`; total supplied $n$ defines support and centered training folds impose feasibility caps |
| Validation | path-only `cv=5`; `cv=None` requests standard five-fold regression CV |
| Selection score | stable package string `"neg_response_standardized_mse"` by default, resolving to the public callable; sklearn scorer names, callables, and `None` accepted |
| Final refit | post-fit `search.refit(X, y, rule=...)` or `search.refit(X, y, n_components=...)` returns a fitted clone without mutating search state |
| Path composition | direct `PiPLSRegression` or `Pipeline` whose final step is `PiPLSRegression` |
| Group handling | path-only keyword `groups` routed to group-aware splitters |
| OOF output | explicit post-fit `search.validation_report(X, y, rule=... or n_components=...)`; reports are returned directly and not attached to search state |
| Dataset namespace | optional immutable container and seeded generators under `pipls.datasets` |
| Real-data input | user-owned explicit reading of `X` and `Y`; no registry, metadata, or loader required for fitting |
| Repository datasets | comma-delimited `X.csv`, `Y.csv`, and documentary `metadata.yaml` |
| Weighting | weighted fitting and general sample-weight routing are intentionally out of scope |
| Repository tests | executable behavior and durable file structure; no pinned living prose or documentary metadata values |
| Historical removals | accepted decisions preserve removal history; tests retain negative assertions only for current public or architectural boundaries |
| Publication assets | downstream repositories pin released `pipls` versions |
| Retired benchmark layer | the former focused synthetic scripts, result contracts, tests, and documentation were removed by Decision 0125 |
| Model standardization | current estimator behavior: fold-local centering and optional scaling, followed by full-training refit |
| Future block-aware scaling | valid long-term product scope, but no accepted API or current implementation phase |
| Model inspection | immutable numerical inspection is implemented; examples render every chart directly with Matplotlib and own all panel and report composition |
| Python compatibility | supported and classified on Python 3.10–3.14; metadata keeps `requires-python = ">=3.10"` without an upper bound |
| Runtime dependencies | `numpy>=1.26,<3`, `scikit-learn>=1.4,<2`, and `joblib>=1.2,<2`; the minimum lines are constrained together on Python 3.10 |
| Compatibility CI | separate minimum, supported-Python, and latest-compatible jobs; every job prints resolved interpreter and runtime dependency versions |
| Distribution validation | `make dist-check` builds once and verifies separate clean wheel and sdist installations with one shared public smoke test |
| Served documentation | user guides and API reference only; `docs/decisions/` remains maintainer history and is excluded from MkDocs |

Additional fixed decisions:

- `"auto"` rank search is deterministic and approximate. It becomes exhaustive only inside the
  final small integer interval. `"optimal"` is exhaustive over the complete admissible range.
- Randomized SVD affects only the initial predictor-matrix decomposition. Response and coupling
  decompositions remain exact.
- interpretable Pi-PLS directions, dilation, rank/solver diagnostics, and the standardized map live
  in the read-only `decomposition_` object; construction matrices remain private and standard
  PLS-style fitted attributes remain top-level.
- `PiPLSSearchCV` is a path evaluator rather than a delegated fitted model. Post-fit `refit()`
  requires exactly one named rule or one component count, returns a fitted clone, and leaves search
  evidence unchanged. `best_*` remains the global configured-score optimum. Output-container
  configuration remains carried by the estimator template and returned clone.
- `PiPLSRegression` is the fixed-model estimator and owns no CV, scoring, or selection results;
  `PiPLSSearchCV` is the search meta-estimator and sole package selection interface.
- Real-data examples use the default path-evaluating `PiPLSSearchCV()` for the path and fit a
  separate fixed model
  after a visible component-path choice. Pulp and Sugarcane use explicit counts, while Tobacco
  uses `one_standard_error_result()` for path annotation. All three use `component_path_`, explicit
  search validation reports, and inspection results directly in memory. All three use direct fixed
  estimators. `best_params_` remains a convenience, not the required user decision.
- Refit coefficients and fitted-model methods are accessed on the direct estimator or pipeline
  returned by `search.refit(...)`. No fitted model is attached to `PiPLSSearchCV`, and coefficients
  are not flattened onto search state when preprocessing may change the feature space.
- OOF results produced after using the same splits for model selection are labeled
  `selection-conditioned`, not unbiased external-test estimates.
- Arbitrary nested meta-estimators and general metadata routing are not supported merely because
  scikit-learn can represent them.
- Repository tests follow `.llm/testing.md`: living handoff, roadmap, and dataset metadata contents
  are reviewed but are not mirrored as fixed phrase or field-value assertions.

## Implemented estimator/search correction and API polish

Decisions 0039 and 0040 are fully implemented:

- `PiPLSRegression` now fits one explicit `(n_components, predictor_rank)` pair;
- it owns no CV, scoring, OOF, or search-result parameters and attributes;
- direct fits warn when $n/r_\pi<3$;
- `PiPLSSearchCV` owns feature probes, candidate folds, conditional path selection, explicit OOF
  reporting from stored split indices, and post-fit full-data refitting through a fresh estimator clone;
- the path supplies the private fold engine with the one warning category it may suppress, while
  unrelated warnings remain visible;
- unused rank-grid construction, solver tracing, duplicate candidate metadata, and OOF rescoring
  have been removed from the private selection layer.
- the complete component path is explicit through `n_components_values="all"`;
- random-state forms, cloning, and estimator-template parameter propagation follow scikit-learn
  conventions;
- the stable package string `"neg_response_standardized_mse"` is the default selection parameter and resolves to the public scorer callable;
- duplicate Pi-PLS-specific fitted aliases are removed in favor of canonical `decomposition_` fields;
- the public guides distinguish best evaluated score from a global surface optimum, explain that
  response-standardized MSE is diagnostic when a nondefault scorer drives selection, and are
  included in source distributions.

The estimator/search correction and final minimality audit are complete. The redundant public
path parameter-prefix control has been removed, supported pipelines infer their terminal Pi-PLS
step, and fixed explicit rank pairs are covered by a focused `GridSearchCV` interoperability test.

## Current standardization boundary and deferred block-aware direction

`PiPLSRegression` currently owns leakage-safe model standardization. Every fit centers `X` and `Y`;
`scale=True` also divides both blocks by safe training-sample standard deviations, while
`scale=False` retains centering. Candidate estimators learn these statistics independently in each
training fold. After rank or path selection, the chosen estimator learns them again from the full
training set supplied to `fit()`. Prediction uses the stored statistics and returns responses in
original units.

This existing behavior is not a future preprocessing feature and must not be externalized into a
one-time transform fitted before cross-validation.

Future development may add block-aware variants of model standardization inside the supported
estimator/model-pipeline fitting boundary. Their exact public placement is not decided. The
direction is preserved, but it is expected months from now and has no current API design.

Until the owner starts a dedicated phase:

- do not add provisional block-scaling classes or public names;
- do not reserve constructor parameters or block semantics;
- do not refactor current code in anticipation of a speculative API;
- keep the fixed numerical core independent from preprocessing while retaining standardization in
  the estimator/model-selection layer;
- require every future learned scaling rule to fit inside its corresponding training fold and the
  final full-training refit.

## Retired benchmark layer

Decision 0125 removes the four focused synthetic benchmark scripts, their dedicated tests, the
public benchmark page, and the normative benchmark contract. Their development-validation purpose
was complete, and they did not define package acceptance thresholds. Numerical, estimator, search,
synthetic-data, and randomized-SVD behavior remain protected by focused package tests. Historical
decision records remain unchanged.

## Legacy dataset licensing review

Decision 0041 closes the companion-analysis dataset inventory for the current repository. The
review applied a strict inclusion rule: public download access is not enough; the exact source
material committed or transformed by `pipls` must carry an explicit license or permission granting
redistribution and adaptation for the repository's general use.

- Corn is intentionally excluded because redistribution permission for the exact source file is not
  sufficiently explicit and the dataset adds little beyond the existing NIR examples.
- The companion materials identify the legacy Steel table as Citrination-processed. The matching
  public source candidate, Citrination dataset 153092, provides neither a documented derivation for
  the exact 267-row table nor an explicit dataset license, and its service terms do not grant
  general redistribution of hosted data. Separately licensed 312-row Steel Strength derivatives are
  different datasets and do not establish rights for the companion table.
- The canonical GPML SARCOS page provides the train and test files and attribution but no dataset
  license or redistribution grant.
- FRED-MD is publicly accessible for research, but current FRED terms reserve rights in the
  compilation and require users to resolve third-party series rights; the repository therefore
  cannot redistribute the historical derived regression table with confidence.

No additional legacy dataset is pending integration. A future real dataset must add a distinct
package-level use case and pass the source-level licensing gate before implementation work begins.

## Recent completed increments

Decisions 0107--0116 are complete and partly superseded by Decision 0137. Path result methods remain
non-mutating stored-row inspection, `PiPLSSearchCV` retains global `best_*` evidence without a
declared final row, and `PiPLSRegression` remains fixed-pair only.

The six owner-authorized simplifications are complete. Search inputs are resolved once;
decomposition inspection trusts validated factor arrays; display-factor $QD$ and prediction
diagnostics are derived from independent state; path-wide metadata and profile selection are not
duplicated; and `PiPLSValidationReport` composes an immutable `PiPLSComponentResult` while
preserving its convenience properties. No package release preparation or Python-package publication
work is authorized.
A follow-up behavior-preserving audit cleanup removes the now-unused private finite-vector
inspection helper and the no-op reassignment of the already validated selected result.

Decision 0117 completes the standard BSD 3-Clause license text, names Vishal Agrawal,
Fritjof Nilsson, and Stefan B. Lindström as package authors and current copyright holders of the
repository-authored code and documentation, and adds public plus machine-readable citation
metadata. The companion manuscript is under revision at *Computers & Chemical Engineering* as
CACE-D-26-00847. Dataset-specific licenses remain authoritative, and no release or publication
claim is implied.

Decision 0118 standardizes every maintained example and documentation figure: rendered method
names use `$\Pi$`-PLS, factor labels use upper-case `$P$` and `$Q$` with lower-case diagonal `$d$`,
tiled factor/latent/prediction figures omit redundant subplot titles, prediction main titles remain
on one line, path/profile y-axes start at zero with an upper limit of at least one, dilation plots
use numeric component ticks without repeating the x-axis label, and dense Tobacco response labels
are rotated explicitly.

Decision 0133 names the configurable-generator truth record `PiPLSRegressionTruth` and keeps
`PiPLSLatentGeometryTruth` for the separate manuscript-aligned generator. The former generic
pre-release truth-class name is not retained as an alias.

Decision 0134 gives public result properties type-revealing names: predictor-rank profiles expose
`selected_result`, validation reports expose `cv_mse_mean`, and their boolean summaries use
`is_selection_conditioned` and `has_complete_oof_coverage`. The former pre-release names are
not retained as aliases.

Decision 0136 makes every maintained ordinary five-fold example and tutorial renderer use
`KFold(n_splits=5, shuffle=True, random_state=0)` explicitly. Path selection, matched PLS
comparison, and selection-conditioned OOF prediction use the same seeded partition within a
workflow. Example 03 retains exhaustive `LeaveOneOut`, for which shuffling is not defined.

Decision 0119 adds `make_pipls_latent_geometry()` and `PiPLSLatentGeometryTruth` as an additive
manuscript-aligned synthetic capability. It implements independent standard-normal latent scores
and loading entries plus independent Gaussian predictor/response noise, with no normalization,
orthonormalization, strength, or observed-scale transformation. Existing configurable generators,
examples, estimator/search behavior, and practical real-data workflows are unchanged.

Decision 0120 makes the companion manuscript the scientific source for the canonical public theory
page. The documentation now includes the retained-subspace projector decomposition, the
cross-covariance response-subspace optimization, the latent least-squares and diagonal relations,
the panoramic interpretation, comparative limiting cases, and the nominal fitted dimension
$(r_\pi+q-h)h$ after $\Pi$ is fixed. It distinguishes retained observed predictor directions from
known synthetic signal rank and leaves every package behavior and practical workflow unchanged.

Decision 0121 establishes the canonical terminology for the fixed construction. $\Pi$ is the
retained predictor basis, $\Pi\Pi^{\mathsf T}$ the retained-subspace projector, $P$ and $Q$
orthonormal predictor and response directions, and $d_k$ the dilation of paired latent mode $k$.
`n_components` counts paired latent modes and `predictor_rank` is the retained predictor-subspace
dimension. `PiPLSDecomposition` exposes `predictor_directions` and
`response_directions`, while the standard estimator attributes `x_rotations_` and `y_rotations_`
remain for PLS-family compatibility. $P$ and $Q$ remain distinct from reconstruction loadings. The
package's $QD$ orientation is the transpose of manuscript-facing $DQ^{\mathsf T}$.

Decision 0122 propagates that vocabulary through the living README, served guides, generated API
introductions, tutorials, example prose, and public source docstrings. It defines the two public
rank names at their owning entry points, uses directions and dilations for the Pi-PLS factors, and
uses mathematically direct decomposition-field names and retains generic component-path
terminology. Decision 0130 later renames the pre-release decomposition fields without changing
numerical behavior.

Decision 0123 adds the focused companion-manuscript synthetic-data guide. It demonstrates the exact
Gaussian latent generator and truth equations, documents the known oracle synthetic dimensions,
and distinguishes distribution-level, seeded-realization, and complete-publication reproduction.
Complete manuscript grids, comparators, and reporting remain downstream, and package search plus
real-data workflows remain unchanged.

Decision 0130 aligns the public immutable decomposition with the canonical mathematics by renaming
its pre-release fields to `predictor_directions` and `response_directions` without compatibility
aliases. Standard PLS-style `x_rotations_` and `y_rotations_` remain fitted estimator attributes and
reference the same read-only arrays.

## Implemented inspect-decide-refit lifecycle

Decision 0137 governs the implemented `PiPLSSearchCV` lifecycle. The search exposes explicit
post-fit full-data refitting:

```python
search = PiPLSSearchCV(cv=cv).fit(X, Y)
path = search.component_path_
profile = search.predictor_rank_profile(n_components=4)
model = search.refit(X, Y, n_components=4)
```

`refit()` accepts exactly one of `rule` and `n_components`, supports `"best_score"`,
`"minimum_cv_mse"`, and `"one_standard_error"`, returns a fitted clone of the direct estimator or
terminal-Pi-PLS pipeline, and leaves the search unchanged. The constructor boolean `refit`, selected
fitted-model attributes, and search-level model delegation were removed in the same increment
because a same-named constructor attribute would shadow the method.

`PiPLSSearchCV.validation_report(X, Y, rule=... or n_components=...)` is now implemented. It uses the
same selected-row resolver as `refit()`, reuses defensive read-only copies of the exact materialized
search splits, always returns ordered OOF predictions and counts, does not rescore candidates or fit
a full-data model, and leaves the search unchanged. The supplied data must have the fitted sample,
feature, and response-column shape and remain in the original row order; the search retains indices,
not values.

The remaining constructor-time selection and OOF controls and selected report state are removed.
The search surface now contains only candidate evidence, immutable path/profile inspection, exact
stored splits for explicit follow-up reporting, and the two post-fit operations.

## Current next increment

The inspect-decide-refit transition and its final presentation cleanup are complete. Maintained
examples and tutorial renderers use `search.refit(...)` rather than manual selected-rank transfer
and obtain selection-conditioned OOF predictions through `search.validation_report(...)` rather
than a separate `cross_val_predict()` pass. The active repository surface contains no
compatibility state from the former lifecycle.

Any release-preparation work requires a separate owner-authorized increment.

Future datasets still require a distinct package-level use case and verified source-level
redistribution rights. Block-aware scaling still requires a separate owner decision.

The representative Pulp, Sugarcane, and Tobacco examples are
complete. Corn, the legacy Steel table, SARCOS, and FRED-MD are intentionally outside the repository
under Decision 0041. The included assets remain validation and documentation material during
release hardening.

The repository-product cleanup is complete. Paper-reproduction repositories remain outside this
roadmap and may independently depend on specific tagged `pipls` releases.

## Authority and drift handling

Use this order when sources disagree:

1. explicit decisions from the project owner in the current request;
2. accepted decision records under `docs/decisions/`;
3. normative `.llm/product_scope.md`, `.llm/mathematics.md`, `.llm/numerical_contracts.md`,
   `.llm/public_api.md`, `.llm/data_io.md`, `.llm/dataset_layout.md`, and
4. source and tests as evidence of implemented behavior;
5. this current-state handoff and `.llm/strategy.md`.

When documentation, tests, and implementation conflict, stop, identify the exact conflict, and
resolve it in the same patch or ask the project owner for a scientific/public-API decision.

## Fresh-chat startup checklist

From an uploaded snapshot, a maintainer should:

1. inspect `.llm/SNAPSHOT_INFO` and confirm whether the snapshot was created from a clean commit;
2. read `.llm/README.md`, this file, `.llm/product_scope.md`, `.llm/strategy.md`, and
   `.llm/project.md`;
3. read `.llm/decisions.md` and the decision records relevant to the requested increment;
4. read the applicable mathematical, numerical, API, and development contracts;
5. inspect the affected source and tests rather than trusting document claims alone;
6. read `.llm/data_io.md` and `.llm/dataset_layout.md` for dataset, real-data, or example work;
8. read `.llm/analysis.md` for fitted-model interpretation, plotting, prediction diagnostics, or
   analysis artifacts;
9. read `.llm/testing.md` before changing repository-document, metadata, or fixture tests;
10. verify that the requested work is the current increment or that the owner explicitly changed
    the order;
11. return a downloadable root-relative unified Git patch, its SHA-256 checksum, validation results, and the concise apply/check/commit/snapshot command sequence.

Routine package work should not require re-uploading a manuscript. Request external scientific
material only when the repository contracts identify a genuine unresolved scientific choice.
