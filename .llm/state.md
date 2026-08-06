# Current development state

## Purpose

This is the fresh-chat handoff for the implemented repository. It records the current package
boundary, accepted exclusions, active maintenance work, and authority order. Historical patch
sequences belong in numbered decisions and Git history, not here.

Read this file with `.llm/product_scope.md` before proposing work. Confirm every relevant claim
against the affected source and tests in the uploaded snapshot.

## Implemented package boundary

Pi-PLS is an installable scikit-learn-compatible package for multivariate regression. The runtime
package contains:

- `PiPLSRegression`, which fits one explicit `(n_components, predictor_rank)` pair;
- `PiPLSSearchCV`, which evaluates the admissible triangular component/rank path;
- immutable component-path, selection, predictor-rank-profile, decomposition, and OOF records;
- public response-standardized MSE scorers;
- immutable datasets, deterministic synthetic generators, and package-owned Pulp, Sugarcane, and
  Tobacco reference resources;
- pure numerical fitted-model inspection under `pipls.inspection`.

The top-level wildcard surface is intentionally narrow:

```python
from pipls import PiPLSRegression, PiPLSSearchCV, PredictorRankSupportWarning
```

Result records and utility functions remain public from their focused modules. There is no
compatibility alias for removed pre-release names.

## Modeling and selection lifecycle

`PiPLSRegression` owns fixed fitting only. It learns centering and optional scaling from the data
supplied to each fit, validates the requested ranks, and exposes standard PLS-style fitted arrays
plus `decomposition_`.

`PiPLSSearchCV.fit(X, y)` materializes one validation split set and evaluates fixed-model clones.
It does not retain `X` or `y` and does not automatically fit a final full-data model. Post-search work
is explicit:

```python
search = PiPLSSearchCV(cv=cv).fit(X, y)
selection = search.select(rule="minimum_cv_mse")
report = search.oof_report(X, y, selection=selection)
model = search.refit(X, y, selection=selection)
```

The implemented named rules are:

- `best_score`: maximum configured-score row on the predictor-rank-conditioned component path;
- `minimum_cv_mse`: smallest conditioned path row satisfying simultaneous relative and absolute
  tolerances around the exact minimum CV-MSE row.

For optimized predictor-rank policies, `PiPLSSearchCV` applies separate constructor-level relative
and absolute configured-score tolerances at every component count. Adaptive refinement and
`rank_test_score` retain private numerical tie semantics. Optimized path rows, selections, and
profiles carry immutable `PiPLSPredictorRankEvidence`; fixed and maximum policies carry none.

Manual selection uses an evaluated `n_components` value and the predictor rank already selected
conditionally for that row. A successful refit attaches the exact immutable row as
`model.selection_`; the fitted search is not mutated. `refit(selection=...)` validates one
pre-existing selection with the same exact compatibility contract used by `oof_report()`, fits the
selected pair, and attaches the exact supplied object after fitting succeeds. Rule-based and manual
component-count refitting remain supported.

`oof_report()` accepts an existing compatible selection and reuses every split materialized by the
search. Repeated validation predictions are averaged per observation and their counts are exposed.
The report is selection-conditioned descriptive validation, not nested-CV or external-test
performance.

## Current statistical reporting

For each evaluated component-path row, validation MSE is summarized with equal weight per
materialized split:

```text
cv_mse_mean = mean(split_cv_mse)
cv_mse_std  = std(split_cv_mse, ddof=0)
```

`cv_mse_std` is descriptive split-to-split variability. Maintained plots use mean CV-MSE plus or
minus one split SD. The package exposes no standard-error result and no standard-error selection
rule.

For `minimum_cv_mse`, `relative_tolerance=None` resolves to
`sqrt(np.finfo(np.float64).eps)` and `absolute_tolerance=np.inf` disables the absolute cap. Both
caps must hold. The selection retains the resolved tolerances, the exact unruled reference minimum,
and the derived effective threshold.

## Data and example boundary

Users may fit ordinary array-like `X` and `y` from any source. No generic registry, downloader,
DataFrame requirement, or package-specific data-ingestion layer is required.

Named reference datasets are available from `pipls.datasets`:

```python
load_pulp()
load_sugarcane()
load_tobacco()
```

Each loader reads one canonical package-resource `X.csv`/`Y.csv` pair plus metadata, README, and
license files. The resources are usable directly outside Python. Corn, the legacy Citrination
Steel table, SARCOS, and FRED-MD are intentionally excluded because the exact candidate materials
do not have sufficiently clear redistribution rights.

The maintained numbered examples are user tasks:

1. compact Pulp automatic fit and fitted-value diagnostic;
2. synthetic inspect-decide-refit workflow with external-test prediction;
3. Pi-PLS versus ordinary-PLS component-path comparison;
4. complete repeated-CV Pulp analysis;
5. complete Sugarcane analysis;
6. complete Tobacco analysis with separate 10% predictor-rank and component-count tolerances.

Example 04 uses `RepeatedKFold(n_splits=5, n_repeats=10, random_state=0)`. Examples 05 and 06 use
`KFold(n_splits=5, shuffle=True, random_state=0)`. Complete real-data examples are exercised by
`make examples`, not duplicated in the default test suite.

## Inspection and rendering boundary

`pipls.inspection` returns validated immutable NumPy results for Pi-PLS factor displays, shared
PLS-family latent structure, balanced biplot coordinates, observation diagnostics, and explicit-
provenance prediction diagnostics.

The runtime package contains no plotting module. Examples and users compose Matplotlib artists,
labels, layouts, saving, and optional `adjustText` placement directly from immutable numerical
results. Generated figures are artifacts, not package state.

## Documentation and distribution boundary

The repository is the long-lived software product, not a paper-reproduction environment. Paper-
specific experiment grids, cached results, figure reproduction, and publication environments
belong in downstream repositories that pin a released package version.

The served site separates tutorials, programming reference, advanced scientific guidance, and
project validation. Numbered decisions and `.llm` are maintainer records and are not served as user
documentation.

Python 3.10--3.14 is supported within the dependency ranges declared in `pyproject.toml`. Clean
wheel and source-distribution installations are validated. Package snapshots are root-relative
archives of a clean committed tree.

## Explicit exclusions and deferred work

Do not add without a new owner decision:

- block-aware scaling or preprocessing semantics;
- automatic outer validation or unbiased-performance claims;
- weighted fitting or general-purpose metadata routing;
- arbitrary nested meta-estimator support;
- public plotting helpers or style objects;
- generic dataset registries, network downloaders, or hidden preprocessing;
- publication-only analyses in this repository;
- compatibility aliases for removed pre-release APIs.

Current centering and optional scaling are not deferred: they are integral to every estimator fit
and are learned within each training fold during search.

## Current maintenance status

Decision 0147 is implemented. Decision records, maintainer context, structural tests, snapshot
policy, and dataset-module ownership are in their normalized current form.

Decision 0148 is implemented. Predictor-rank tolerances, conditioned component-path selection,
immutable rank evidence, and the separate Tobacco predictor-rank and component-count demonstrations
are part of the current package contract.

Decision 0149 is implemented. The runtime API, maintained examples, public documentation,
retained decisions, tests, and distribution checks use `search_method="adaptive"` and
`search_method="exhaustive"`; the former values are rejected without aliases.

Decision 0150 is implemented. The served computational-performance guide documents candidate-fit
scaling, validation repetitions, adaptive and exhaustive rank coverage, fixed and restricted
paths, randomized predictor SVD, parallel execution, OOF reuse, and work inspection. Public guides
route to it, and source-distribution documentation validation protects its shipped and rendered
forms.

Decision 0151 is implemented. `refit(selection=...)` shares exact compatibility validation with
`oof_report(selection=...)`; evidence-retaining examples and guides pass one pre-refit selection
through search inspection, optional OOF inspection, and final fitting. Every served tutorial has
one source-level Mermaid flowchart with equivalent prose, and strict local, inherited Pages-overlay,
and source-distribution checks protect diagram rendering. The former presentation increment from
Decision 0139 is superseded.

Decision 0152 is implemented. Manual component-count tutorials first inspect an unselected path,
then choose the component count and create one selection, then review the selected path and
conditional rank or OOF evidence. Their diagrams contain one possible return to the selection
step. Same-search OOF results are descriptive selection-conditioned evidence rather than
independent qualification. Tutorials 2 and 3 retain separate unselected and selected path
artifacts.

Decision 0153 is implemented. Package-owned protocol detection, provenance, examples, support
claims, and dedicated tests are removed. The maintained example catalogue is contiguous from 01
through 06. Generic splitter interoperability, ordered OOF reporting, partial and repeated coverage,
pooled OOF $R^2$, and protocol-neutral singleton-validation scorer safety remain. Current decisions
and the retirement map use this completed boundary.

Decision 0154 is accepted. A seven-patch cleanup is replacing tests that police repository prose or
source structure with behavioral, machine-readable artifact, executable-tool, and installed-
distribution coverage. Patches 1--6 are complete. Pure documentation, decision, repository,
workflow, configuration-literal, compatibility-policy, example-source, AST-workflow, snippet-
marker, plotting-literal, SVG-text, and historical-name tombstone tests are removed. Artifact
validation shares private subprocess, virtual-environment, archive, and artifact helpers; installed
and source-distribution checks execute checked-in behavior; public-result tests assert current fields
rather than removed aliases; and overlapping selection, OOF, pickle, and search-nonmutation tests are
consolidated. The Pulp example and tutorial renderer now delegate figure construction and manifest
writing to caller-local helpers while preserving analytical order and byte-identical tutorial
artifacts. Runtime behavior and the installed package surface are unchanged.

## Authority and drift handling

When sources disagree, use this order:

1. an explicit owner instruction in the current request;
2. accepted numbered decisions;
3. normative `.llm` contracts, especially `product_scope.md`, `mathematics.md`,
   `numerical_contracts.md`, `public_api.md`, `data_io.md`, and `dataset_layout.md`;
4. source and tests as evidence of implemented behavior;
5. this handoff and `.llm/strategy.md`.

Do not silently choose between conflicting scientific or public-API contracts. Identify the exact
conflict and resolve it in the same patch or obtain an owner decision.

## Fresh-chat checklist

1. Inspect `.llm/SNAPSHOT_INFO` and confirm the clean source commit.
2. Read `.llm/README.md`, this file, `.llm/product_scope.md`, and `.llm/strategy.md`.
3. Read `.llm/decisions.md` and every decision directly relevant to the requested change.
4. Read the applicable mathematical, numerical, API, data, analysis, testing, and development
   contracts.
5. Inspect affected source and tests before editing.
6. Verify that the request fits accepted scope and identify any decision that governs it.
7. Return one root-relative Git patch, its SHA-256 checksum, validation evidence, and concise
   apply/check/commit/snapshot commands.
