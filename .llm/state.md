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

Phases A through E4c are complete and committed. The first broad E4 benchmark implementation was removed and replaced by focused question-specific benchmarks:

- repository, packaging, deterministic root-relative snapshots, and direct Git patch workflow;
- fixed-parameter Pi-PLS numerical core;
- scikit-learn-compatible fixed-model `PiPLSRegression` for one explicit
  `(n_components, predictor_rank)` pair;
- independent full, randomized, and automatic predictor-SVD policies;
- hardened public validation and `StatisticalSupportWarning` for direct fixed fits with
  fewer than four observations per retained predictor-rank direction;
- pipeline-aware `PiPLSPathCV` for triangular `(n_components, predictor_rank)` search;
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
- a current reference suite containing pulp, sugarcane, and tobacco.

The current top-level package exports are:

```python
from pipls import (
    PiPLSDecomposition,
    PiPLSPathCV,
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
- user documentation and concise executable examples;
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
| Path search | `PiPLSPathCV(search_method="auto")` by default |
| Component-path artifact | `component_path_results_`: one row per component count with numeric predictor rank, policy, mean CV-MSE, fold SD, and split count |
| Exhaustive search | explicit `PiPLSPathCV(search_method="optimal")` |
| Predictor SVD | `svd_solver="auto"`, with the documented conservative threshold |
| Reproducibility | `random_state=0` by default |
| Rank support rule | path-only `samples_per_predictor_rank=5`; total supplied $n$ defines support and centered training folds impose feasibility caps |
| Validation | path-only `cv=5`; `cv=None` requests standard five-fold regression CV |
| Selection score | response-standardized negative MSE by default; `scoring=None` uses estimator score |
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

Additional fixed decisions:

- `"auto"` rank search is deterministic and approximate. It becomes exhaustive only inside the
  final small integer interval. `"optimal"` is exhaustive over the complete admissible range.
- Randomized SVD affects only the initial predictor-matrix decomposition. Response and coupling
  decompositions remain exact.
- `PiPLSRegression` is the fixed-model estimator and owns no CV, scoring, or selection results;
  `PiPLSPathCV` is the path meta-estimator and sole package selection interface.
- Real-data examples use `PiPLSPathCV(refit=False)` for the path, treat CSV as canonical,
  derive PDFs from the CSV, and fit a separate fixed `PiPLSRegression` after an explicit component
  choice. `best_params_` remains a convenience, not the required user decision.
- Path coefficients are accessed through `best_pipls_` or `best_estimator_`; they are not flattened
  onto `PiPLSPathCV` when preprocessing may change the feature space.
- OOF results produced after using the same splits for model selection are labeled
  `selection-conditioned`, not unbiased external-test estimates.
- Arbitrary nested meta-estimators and general metadata routing are not supported merely because
  scikit-learn can represent them.
- Repository tests follow `.llm/testing.md`: living handoff, roadmap, and dataset metadata contents
  are reviewed but are not mirrored as fixed phrase or field-value assertions.

## Accepted staged estimator/search correction

Decision 0039 is partially implemented. Patches 2 through 4 established and consolidated the estimator/search boundary:

- `PiPLSRegression` now fits one explicit `(n_components, predictor_rank)` pair;
- it owns no CV, scoring, OOF, or search-result parameters and attributes;
- direct fits warn when $n/r_\pi<4$;
- `PiPLSPathCV` owns feature probes, candidate folds, conditional path selection, optional OOF
  fitting, and selected full-data refitting;
- the path supplies the private fold engine with the one warning category it may suppress, while
  unrelated warnings remain visible;
- unused rank-grid construction, solver tracing, duplicate candidate metadata, and OOF rescoring
  have been removed from the private selection layer.

The example and guide alignment is complete; the final minimality audit remains.

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
duplicated examples 10–12. Pulp, Sugarcane, and Tobacco remain transparent component-path examples,
and `make examples` runs every numbered example as an explicit application-validation action.
They write separate Pi-PLS and standard PLS (NIPALS) path CSVs for the same folds and component
counts, call imported helpers to generate an overlaid PDF from those tables, expose a visible user
component choice, and fit a separate fixed `PiPLSRegression` with both Pi-PLS ranks recorded
explicitly. The examples contain no subprocess wrappers or repeated table-validation boilerplate.
Default tests retain dataset-layout, component-path API, PLS-helper, plotting, and workflow-structure
contracts without executing the complete real-data analyses.

Every benchmark owns one readable script and one minimal CSV output. Generated CSV files remain
ignored and are excluded from snapshots. Software versions, execution controls, timings, and
unrelated metrics are omitted unless they answer that benchmark's explicit question. OLS, CCA,
publication grids, and figure generation remain outside the repository.

## Current next increment

Complete Decision 0039 with patch 6/6: perform the final API and minimality audit. Do not add new
features, datasets, benchmarks, or block-scaling API during that audit.

After the estimator/search correction and final audit, resume user documentation and release
hardening. Do not add another dataset or benchmark without a new package-level question.

## Subsequent roadmap

1. **Fixed-estimator/path-search correction:** implement Decision 0039 through fixed estimator, sole
   path selection, private-code consolidation, example alignment, and final audit patches.
2. **User documentation and release hardening:** buildable user guide, API reference, compatibility
   policy, packaging checks, and versioned releases.
3. **Future product development:** additional estimators, validation tools, datasets, and—only after
   a separate owner decision—block-aware variants of the existing model-internal standardization.

The focused synthetic benchmarks and representative Pulp, Sugarcane, and Tobacco examples are
complete. They remain validation and documentation assets during the architectural correction.

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
8. read `.llm/testing.md` before changing repository-document, metadata, or fixture tests;
9. verify that the requested work is the current increment or that the owner explicitly changed
   the order;
10. return one root-relative unified Git patch, validation results, and exact direct Git commands.

Routine package work should not require re-uploading a manuscript. Request external scientific
material only when the repository contracts identify a genuine unresolved scientific choice.
