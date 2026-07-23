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
context-free API demonstrations. Example 07 has been removed; example 02 now supplies the short
synthetic path-selection tutorial and independent-test prediction workflow.

Phases A through F4 are complete and committed. The first broad E4 benchmark
implementation was removed and replaced by focused question-specific benchmarks:

- repository, packaging, deterministic root-relative snapshots, and direct Git patch workflow;
- fixed-parameter Pi-PLS numerical core;
- scikit-learn-compatible fixed-model `PiPLSRegression` for one explicit
  `(n_components, predictor_rank)` pair;
- independent full, randomized, and automatic predictor-SVD policies;
- hardened public validation and `StatisticalSupportWarning` for direct fixed fits with
  fewer than three observations per retained predictor-rank direction;
- transactional fixed and path fits, range-safe boundary preprocessing, safe read-only or
  overlapping `copy=False` inputs, and finite public fitted/output values;
- pipeline-aware `PiPLSPathCV` for triangular `(n_components, predictor_rank)` search;
- immutable `PiPLSComponentPath` arrays and frozen scalar lookup through `component_path_`;
- on-demand immutable `PiPLSPredictorRankProfile` results through
  `predictor_rank_profile(n_components)`, derived from `cv_results_`;
- a literal-matrix first example showing one fixed fit, prediction, and decomposition plot without CV;
- a public documentation entry that defines paired latent variables, component-count scanning,
  and elbow-based CV-MSE interpretation before specialized terminology;
- shared private fold-evaluation and adaptive-search machinery;
- PLS-style fitted attributes, feature names, pandas output, inverse reconstruction, and public
  immutable `PiPLSDecomposition`;
- grouped, repeated, predefined, temporal, and leave-one-out split workflows;
- optional ordered OOF predictions and immutable `PiPLSValidationReport` with explicit
  fixed-parameter versus selection-conditioned labeling;
- immutable validated `PiPLSDataset` and deterministic synthetic generators with shared,
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
- a two-tier tutorial route: a short deterministic synthetic selection-and-prediction workflow
  from example 02, followed by a focused Pulp real-data workflow from example 10; both use
  checked snippets and deterministic single-chart SVG assets;
- direct Tobacco response pagination through two caller-owned multipage PDFs, with full predictor
  SVD, decreasing-wavenumber spectral axes, and raw observation diagnostics;
- a direct Pi-PLS/ordinary-PLS comparison whose two immutable component paths are plotted together
  in example 09 without DataFrame conversion, generated CSV intermediates, or a plotting helper;
- a tutorial-first served site whose synthetic entry tutorial owns the minimum selection contract
  and whose second Pulp tutorial owns real-data selection qualification, selection-conditioned
  OOF analysis, and representative interpretation plots; both link to the generated API and
  advanced references;
- a compact documentation navigation without separate quickstart, estimator, parameter-selection,
  or preprocessing guides: the home page provides the minimal fixed fit, generated API pages own
  exact estimator contracts, and retained path and validation pages cover advanced behavior;
- an audience-oriented documentation entry: the root README owns package orientation, installation,
  two compact workflows, and tutorial routes; `CONTRIBUTING.md` owns development and repository
  maintenance; served navigation separates programming reference from project validation;
- a completed documentation reference layer with a public result-object map, task-oriented
  troubleshooting, generic local-link and anchor validation, and tests that protect structure
  without freezing explanatory prose.

Decisions 0079--0083 establish and enforce data-first rendering for biplots, prediction
diagnostics, standard PLS-family inspection, and Pi-PLS factors. Plotting migration G1--G5 is
complete. First-release preparation is the next increment. Decision 0042 defines the staged fitted-model architecture, and Decision 0045 corrects the
analysis-model boundary. The shared API and numbered-example migrations are complete. Ordinary PLS
is retained only in the dedicated example-09 CV-MSE comparisons and declared benchmarks. Examples
10–12 evaluate Pi-PLS paths only, and every post-analysis quantity comes from the selected Pi-PLS
model.

The current top-level package exports are:

```python
from pipls import (
    PiPLSComponentPath,
    PiPLSComponentResult,
    PiPLSDecomposition,
    PiPLSPathCV,
    PiPLSPredictorRankProfile,
    PiPLSRegression,
    PiPLSValidationReport,
    StatisticalSupportWarning,
)
```

Dataset functionality is public from `pipls.datasets`:

```python
from pipls.datasets import (
    PiPLSDataset,
    PiPLSSyntheticTruth,
    make_pipls_regression,
    make_pipls_train_test,
)
```

## Accepted repository-product boundary

Decision 0024 defines `pipls` as a long-lived software-product repository. It owns:

- the installable package and public API;
- user documentation and numbered examples from a minimal fixed fit through complete analyses;
- transparent reference datasets;
- lightweight synthetic and real-data validation benchmarks;
- tests, packaging, compatibility policy, and releases.

It does not own manuscript figures, complete publication experiment grids, paper-only OLS/CCA
comparisons, cached paper results, or publication-specific environments. Those belong in downstream
reproduction repositories that pin tagged `pipls` releases.

Ordinary PLS remains an appropriate package benchmark because it is the nearest practical baseline
for Pi-PLS users. OLS or CCA are included only when they protect a package-level identity, limiting
case, or public behavior.

## Accepted public defaults and boundaries

| Concern | Current contract |
|---|---|
| Conditional predictor-rank selection | `PiPLSPathCV(search_method="auto")` |
| Component counts | `n_components_values="all"` by default; explicit integer sequences request a subset |
| Path search | `PiPLSPathCV(search_method="auto")` by default |
| Component-path artifact | immutable `component_path_` with aligned score, CV-MSE, fold-SD, predictor-rank, policy, and split-count arrays plus scalar lookup |
| Conditional rank profile | `predictor_rank_profile(h)` returns evaluated ranks and aligned score/CV-MSE arrays on demand, plus the selected scalar row |
| Exhaustive search | explicit `PiPLSPathCV(search_method="optimal")` |
| Predictor SVD | `svd_solver="auto"`, with the documented conservative threshold |
| Reproducibility | estimator `random_state` accepts integer, NumPy `RandomState`, or `None`; default `0` is reproducible |
| Rank support rule | path-only `samples_per_predictor_rank=5`; total supplied $n$ defines support and centered training folds impose feasibility caps |
| Validation | path-only `cv=5`; `cv=None` requests standard five-fold regression CV |
| Selection score | public response-standardized negative-MSE callable by default; sklearn scorer names, callables, and `None` accepted |
| Path composition | direct `PiPLSRegression` or `Pipeline` whose final step is `PiPLSRegression` |
| Group handling | path-only keyword `groups` routed to group-aware splitters |
| OOF output | path-only opt-in through `return_oof_predictions=True` |
| Dataset namespace | optional immutable container and seeded generators under `pipls.datasets` |
| Real-data input | user-owned explicit reading of `X` and `Y`; no registry, metadata, or loader required for fitting |
| Repository datasets | comma-delimited `X.csv`, `Y.csv`, and documentary `metadata.yaml` |
| Weighting | weighted fitting and general sample-weight routing are intentionally out of scope |
| Repository tests | executable behavior and durable file structure; no pinned living prose or documentary metadata values |
| Publication assets | downstream repositories pin released `pipls` versions |
| Synthetic benchmark plan | one user question, one readable script, and one minimal CSV output per benchmark; ordinary PLS is the sole planned external comparator |
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
- Pi-PLS-specific factorization and numerical diagnostics live only in the read-only
  `decomposition_` object; standard PLS-style fitted attributes remain top-level.
- Refit-dependent path methods are absent when `refit=False`; output-container configuration is
  carried by the estimator template rather than a second path-level `set_output` layer.
- `PiPLSRegression` is the fixed-model estimator and owns no CV, scoring, or selection results;
  `PiPLSPathCV` is the path meta-estimator and sole package selection interface.
- Real-data examples use `PiPLSPathCV(refit=False)` for the path and fit a separate fixed model
  after an explicit component choice. Pulp, Sugarcane, and Tobacco use `component_path_`,
  scikit-learn OOF prediction, and inspection results directly in memory. All three use direct fixed
  estimators. `best_params_` remains a convenience, not the required user decision.
- Path coefficients are accessed through `best_pipls_` or `best_estimator_`; they are not flattened
  onto `PiPLSPathCV` when preprocessing may change the feature space.
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
- `PiPLSPathCV` owns feature probes, candidate folds, conditional path selection, optional OOF
  fitting, and selected full-data refitting;
- the path supplies the private fold engine with the one warning category it may suppress, while
  unrelated warnings remain visible;
- unused rank-grid construction, solver tracing, duplicate candidate metadata, and OOF rescoring
  have been removed from the private selection layer.
- the complete component path is explicit through `n_components_values="all"`;
- random-state forms and refit-dependent method availability follow scikit-learn conventions;
- the public response-standardized scorer callable is the default selection metric;
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

## Focused benchmark status

Decision 0030 supersedes the earlier universal manifest, universal result schema, and broad CI
runner. Those implementation assets have been removed. The four focused synthetic questions are
implemented independently:

1. fixed-structure recovery by fixed Pi-PLS;
2. adaptive Pi-PLS rank selection;
3. paired Pi-PLS versus ordinary PLS prediction under predictor-specific nuisance;
4. full-versus-randomized solver consistency.

The earlier real-data smoke-check scripts and full example-execution tests were removed because they
duplicated the numbered analyses. `make examples` runs every numbered example as an explicit
application-validation action. Example 09 keeps immutable Pi-PLS and standard PLS (NIPALS) paths
in memory and generates the overlaid comparison PDFs directly. Examples 10–12 plot
`component_path_` without table conversion, fit selected fixed estimators, use
scikit-learn `cross_val_predict`, calculate inspection results in memory, and write only final PDF
figures. Pulp additionally plots the evaluated predictor-rank profile at three components. Tobacco
writes three-page prediction-diagnostic and coefficient PDFs while preserving source-order response
pagination. Required result directories are tracked and preserved by `make clean`. Default tests
retain dataset-layout, component-path API, immutable PLS-helper, and workflow-structure contracts
without executing the artifact-producing real-data scripts.

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

Every benchmark owns one readable script and one minimal CSV output. Generated CSV files remain
ignored and are excluded from snapshots. Software versions, execution controls, timings, and
unrelated metrics are omitted unless they answer that benchmark's explicit question. OLS, CCA,
publication grids, and figure generation remain outside the repository.

## Current next increment

First-release preparation: choose the initial version, complete metadata and release notes, and
rehearse the tag and publication checklist.

## Subsequent roadmap

1. **First-release preparation:** choose the initial version, complete metadata and release
   notes, and rehearse the tag and publication checklist.
2. **First tagged release:** publish only after the rehearsal and checklist pass.

Future datasets still require a distinct package-level use case and verified source-level
redistribution rights. Block-aware scaling still requires a separate owner decision.

The focused synthetic benchmarks and representative Pulp, Sugarcane, and Tobacco examples are
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
   `.llm/benchmarking.md`;
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
7. read `.llm/benchmarking.md` for benchmark questions, scripts, metrics, outputs, or fixtures;
8. read `.llm/analysis.md` for fitted-model interpretation, plotting, prediction diagnostics, or
   analysis artifacts;
9. read `.llm/testing.md` before changing repository-document, metadata, or fixture tests;
10. verify that the requested work is the current increment or that the owner explicitly changed
    the order;
11. return one root-relative unified Git patch, validation results, and exact direct Git commands.

Routine package work should not require re-uploading a manuscript. Request external scientific
material only when the repository contracts identify a genuine unresolved scientific choice.
