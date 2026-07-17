# Current development state

## Purpose

This is the concise handoff document for starting work from a repository snapshot without prior
chat history. It records the current implemented boundary, accepted owner decisions, deferred
scope, and next admissible increment. Update it whenever a patch changes a phase, public default,
supported composition boundary, or roadmap order.

A new chat should read this file before proposing implementation work. Do not infer current state
from an earlier conversation, an old patch, or superseded planning material alone.

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
- a transparent real-data input contract: users and examples read `X` and `Y` explicitly,
  with no metadata, registry, or package-owned loader required for fitting;
- a repository real-dataset convention using comma-delimited `X.csv`, `Y.csv`, and documentary
  `metadata.yaml`.

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
| Dataset namespace | optional immutable container and seeded generators under `pipls.datasets` |
| Real-data input | user-owned explicit reading of `X` and `Y`; no registry, metadata, or loader required for fitting |
| Repository datasets | comma-delimited `X.csv`, `Y.csv`, and documentary `metadata.yaml` |
| Weighting | weighted fitting and general sample-weight routing are intentionally out of scope |
| Repository tests | executable behavior and durable file structure; no pinned living prose or documentary metadata values |

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
- Repository tests follow `.llm/testing.md`: living handoff, roadmap, and dataset metadata
  contents are reviewed but are not mirrored as fixed phrase or field-value assertions.

## Current next increment

Phase E3 is complete for the initial reference suite. The repository contains four transparent
integrations: BSD-licensed Linnerud and CC BY 4.0 pulp, sugarcane, and tobacco. All use `X.csv`,
`Y.csv`, and `metadata.yaml`; their examples read only the two comma-delimited model tables
explicitly. Tobacco contributes 347 samples, 1,557 raw FT-NIR predictors, and 13 responses with
public ID alignment and no spectral preprocessing.

The next implementation patch should begin **Phase E4: benchmark fixtures**. Define a small,
deterministic, reviewable benchmark manifest spanning the four integrated datasets and fixed public
estimator configurations. Keep benchmark expectations separate from manuscript-result claims and
from living dataset metadata. Establish update rules, tolerances, runtime limits, and the exact
metrics before freezing any numerical fixture. Do not add another dataset in the same increment.

Corn remains deferred until its unresolved preprocessing choices are fixed. When Corn is added,
expose its public raw-data reading and analysis-relevant preprocessing directly. Do not claim
manuscript reproduction from the current reference examples.

## Subsequent roadmap

- **E3 — real dataset integrations:** complete for the initial Linnerud, pulp, sugarcane, and
  tobacco reference suite.
- **E4 — benchmark fixtures:** deterministic benchmark expectations and regression tolerances
  linking synthetic and migrated datasets to estimator/path behavior.
- **F1 — paper reproduction:** scripts and manifests for figures, tables, and paper-specific rank
  and validation rules; scripts still read `X` and `Y` visibly.
- **F2 — release hardening:** clean-install/build tests, frozen reproduction tolerances, metadata,
  licensing audit, and release documentation.
- **F3 — archival release:** tagged release, DOI/archive workflow, and manuscript repository
  reference.

The project owner may reorder these increments explicitly. A patch must not silently reorder them.

## Authority and drift handling

Use this order when sources disagree:

1. explicit decisions from the project owner in the current request;
2. accepted decision records under `docs/decisions/`;
3. normative `.llm/mathematics.md`, `.llm/numerical_contracts.md`, `.llm/public_api.md`,
   `.llm/data_io.md`, and `.llm/dataset_layout.md`;
4. source and tests as evidence of implemented behavior;
5. this current-state handoff and `.llm/strategy.md`.

When documentation, tests, and implementation conflict, stop, identify the exact conflict, and
resolve it in the same patch or ask the project owner for a scientific/public-API decision.

## Fresh-chat startup checklist

From an uploaded snapshot, a maintainer should:

1. inspect `.llm/SNAPSHOT_INFO` and confirm whether the snapshot was created from a clean commit;
2. read `.llm/README.md`, this file, `.llm/strategy.md`, and `.llm/project.md`;
3. read `.llm/decisions.md` and the decision records relevant to the requested increment;
4. read the applicable mathematical, numerical, API, and development contracts;
5. inspect the affected source and tests rather than trusting document claims alone;
6. read `.llm/data_io.md` and `.llm/dataset_layout.md` for dataset, real-data, example, or
   reproduction work;
7. read `.llm/testing.md` before changing repository-document, metadata, or fixture tests;
8. verify that the requested work is the current increment or that the owner explicitly changed
   the order;
9. return one root-relative unified Git patch, validation results, and exact direct Git commands.

Routine work should not require re-uploading the manuscript. Request external scientific material
only when the repository contracts identify a genuine unresolved scientific choice.
