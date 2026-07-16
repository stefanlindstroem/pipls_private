# Current development state

## Purpose

This is the concise handoff document for starting work from a repository snapshot without prior
chat history. It records the current implemented boundary, accepted owner decisions, deferred
scope, and next admissible increment. Update it whenever a patch changes a phase, public default,
supported composition boundary, or roadmap order.

A new chat should read this file before proposing implementation work. Do not infer current state
from an earlier conversation, an old patch, or the historical publication plan alone.

## Implemented boundary

Phases A through E1 are complete and committed:

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
  predictor-specific, and response-specific latent structure.

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
| Dataset namespace | immutable container and seeded generators under `pipls.datasets` |
| Weighting | weighted fitting and general sample-weight routing are intentionally out of scope |

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

## Current next increment

The next implementation patch is **Phase E2: dataset registry and generic loader**.

E2 acceptance conditions:

1. define one versioned registry-entry schema for names, local paths, shapes, names, checksums,
   provenance, licensing, and preparation metadata;
2. validate the tracked `datasets/registry.yaml` before resolving any entry;
3. provide a loader that returns `PiPLSDataset` and supports an explicit alternate data root or
   direct path;
4. verify sample alignment, numeric model columns, finite values, names, provenance, and checksums;
5. perform no implicit download, converter execution, row filtering, imputation, centering,
   scaling, or feature engineering;
6. keep dataset preparation under `scripts/prepare_data/`, outside the installed runtime loader;
7. add schema, invalid-entry, checksum, path-resolution, and no-transformation tests without
   migrating a real research dataset.

Do not begin real dataset conversion, downloading, or paper reproduction in E2.

## Subsequent roadmap

- **E3 — real dataset migrations:** migrate and validate one dataset per coherent increment,
  including the Corn reconstruction only after preprocessing choices are fixed.
- **E4 — benchmark fixtures:** deterministic benchmark manifests and regression tolerances linking
  synthetic and migrated datasets to estimator/path behavior.
- **F1 — paper reproduction:** scripts and manifests for figures, tables, and paper-specific rank
  and validation rules.
- **F2 — release hardening:** clean-install/build tests, frozen reproduction tolerances, metadata,
  licensing audit, and release documentation.
- **F3 — archival release:** tagged release, DOI/archive workflow, and manuscript repository
  reference.

The project owner may reorder these increments explicitly. A patch must not silently reorder them.

## Authority and drift handling

Use this order when sources disagree:

1. explicit decisions from the project owner in the current request;
2. accepted decision records under `docs/decisions/`;
3. normative `.llm/mathematics.md`, `.llm/numerical_contracts.md`, and `.llm/public_api.md`;
4. source and tests as evidence of implemented behavior;
5. this current-state handoff and `.llm/strategy.md`;
6. `docs/publication_repository_plan.md` as the broad historical architecture and rationale.

The publication plan contains proposals that were narrowed or superseded during implementation.
Do not revive them without checking this file and the accepted decisions. When documentation,
tests, and implementation conflict, stop, identify the exact conflict, and resolve it in the same
patch or ask the project owner for a scientific/public-API decision.

## Fresh-chat startup checklist

From an uploaded snapshot, a maintainer should:

1. inspect `.llm/SNAPSHOT_INFO` and confirm whether the snapshot was created from a clean commit;
2. read `.llm/README.md`, this file, `.llm/strategy.md`, and `.llm/project.md`;
3. read `.llm/decisions.md` and the decision records relevant to the requested increment;
4. read the applicable mathematical, numerical, API, and development contracts;
5. inspect the affected source and tests rather than trusting document claims alone;
6. verify that the requested work is the current increment or that the owner explicitly changed
   the order;
7. return one root-relative unified Git patch, validation results, and exact direct Git commands.

Routine work should not require re-uploading the manuscript. Request external scientific material
only when the repository contracts identify a genuine unresolved scientific choice.
