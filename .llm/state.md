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

Phases A through E3 are complete and committed:

- repository, packaging, deterministic root-relative snapshots, and direct Git patch workflow;
- fixed-parameter Pi-PLS numerical core;
- scikit-learn-compatible `PiPLSRegression` with fixed, rule-derived, exhaustive, and adaptive
  predictor-rank modes;
- independent full, randomized, and automatic predictor-SVD policies;
- hardened public validation and `StatisticalSupportWarning` for
  `samples_per_predictor_rank < 5` in rule-based modes;
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
- an initial reference suite containing Linnerud, pulp, sugarcane, and tobacco.

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
| Conditional predictor-rank selection | `PiPLSRegression(predictor_rank="auto")` by default |
| Path search | `PiPLSPathCV(search_method="auto")` by default |
| Exhaustive search | explicit `"optimal"` in either public interface |
| Predictor SVD | `svd_solver="auto"`, with the documented conservative threshold |
| Reproducibility | `random_state=0` by default |
| Validation | `cv=5`; `cv=None` requests standard five-fold regression CV |
| Selection score | response-standardized negative MSE by default; `scoring=None` uses estimator score |
| Path composition | direct `PiPLSRegression` or `Pipeline` whose final step is `PiPLSRegression` |
| Group handling | keyword-only `groups` routed to group-aware splitters |
| OOF output | opt-in through `return_oof_predictions=True` |
| Dataset namespace | optional immutable container and seeded generators under `pipls.datasets` |
| Real-data input | user-owned explicit reading of `X` and `Y`; no registry, metadata, or loader required for fitting |
| Repository datasets | comma-delimited `X.csv`, `Y.csv`, and documentary `metadata.yaml` |
| Weighting | weighted fitting and general sample-weight routing are intentionally out of scope |
| Repository tests | executable behavior and durable file structure; no pinned living prose or documentary metadata values |
| Publication assets | downstream repositories pin released `pipls` versions |
| Future preprocessing | valid long-term product scope, but no accepted API or current implementation phase |

Additional fixed decisions:

- `"auto"` rank search is deterministic and approximate. It becomes exhaustive only inside the
  final small integer interval. `"optimal"` is exhaustive over the complete admissible range.
- Randomized SVD affects only the initial predictor-matrix decomposition. Response and coupling
  decompositions remain exact.
- `PiPLSRegression` is the fixed-model estimator and `PiPLSPathCV` is the path meta-estimator.
  Neither wraps the other; both use shared private search machinery.
- Path coefficients are accessed through `best_pipls_` or `best_estimator_`; they are not flattened
  onto `PiPLSPathCV` when preprocessing may change the feature space.
- OOF results produced after using the same splits for model selection are labeled
  `selection-conditioned`, not unbiased external-test estimates.
- Arbitrary nested meta-estimators and general metadata routing are not supported merely because
  scikit-learn can represent them.
- Repository tests follow `.llm/testing.md`: living handoff, roadmap, and dataset metadata contents
  are reviewed but are not mirrored as fixed phrase or field-value assertions.

## Explicitly deferred preprocessing direction

Future development may add standardization pipelines and block-scaling functionality. The direction
is preserved, but it is expected months from now and has no current API design.

Until the owner starts a dedicated phase:

- do not add provisional preprocessing classes or public names;
- do not reserve constructor parameters or block semantics;
- do not refactor current code in anticipation of a speculative API;
- keep the fixed numerical core independent from preprocessing;
- fit any existing learned preprocessing inside its corresponding training fold.

## Current next increment

The next patch is a **repository-product cleanup**, not a runtime feature patch. It should:

- remove the transitional `paper/` and `scripts/reproduce_paper/` placeholders;
- rewrite public repository navigation around package users rather than one publication;
- remove remaining promises that paper figures, complete comparison grids, or manuscript
  orchestration will be implemented inside `pipls`;
- retain historical scientific context and accepted algorithm decisions where they remain useful;
- make no estimator, dataset, benchmark-result, or preprocessing API change.

After that cleanup, the next implementation phase should define a lightweight **synthetic benchmark
contract** before freezing numerical results. Synthetic validation should use known latent truth and
compare Pi-PLS primarily with ordinary PLS under controlled structures. Real-data validation comes
after the synthetic contract.

Corn remains deferred until its unresolved preprocessing choices are fixed. When Corn is added,
expose its public raw-data reading and analysis-relevant preprocessing directly.

## Subsequent roadmap

1. **Repository-product cleanup:** remove publication placeholders and rewrite public navigation.
2. **Synthetic benchmark contract:** define scenarios, metrics, seeds, tolerances, runtime tiers,
   result schema, and update policy without yet turning the repository into a paper experiment
   system.
3. **Lightweight benchmark implementation:** deterministic synthetic fixtures first, followed by
   representative real-dataset smoke checks.
4. **User documentation and release hardening:** buildable user guide, API reference, compatibility
   policy, packaging checks, and versioned releases.
5. **Future product development:** additional estimators, validation tools, datasets, and—only after
   a separate owner decision—preprocessing, standardization, or block scaling.

Paper-reproduction repositories are outside this roadmap. They may be created independently for one
or several publications and should depend on specific tagged `pipls` releases.

## Authority and drift handling

Use this order when sources disagree:

1. explicit decisions from the project owner in the current request;
2. accepted decision records under `docs/decisions/`;
3. normative `.llm/product_scope.md`, `.llm/mathematics.md`, `.llm/numerical_contracts.md`,
   `.llm/public_api.md`, `.llm/data_io.md`, and `.llm/dataset_layout.md`;
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
7. read `.llm/testing.md` before changing repository-document, metadata, or fixture tests;
8. verify that the requested work is the current increment or that the owner explicitly changed
   the order;
9. return one root-relative unified Git patch, validation results, and exact direct Git commands.

Routine package work should not require re-uploading a manuscript. Request external scientific
material only when the repository contracts identify a genuine unresolved scientific choice.
