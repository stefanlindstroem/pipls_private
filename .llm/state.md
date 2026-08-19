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
profiles carry immutable `PiPLSPredictorRankEvidence`; fixed and EPV policies carry none.

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
6. complete Tobacco analysis with separate 10% predictor-rank and component-count tolerances;
7. matched-split Pulp comparison of the peer-reviewed cross-covariance and optional least-squares
   response-subspace policies, without final refitting.

Example 04 uses `RepeatedKFold(n_splits=5, n_repeats=10, random_state=0)`. Examples 05, 06, and 07
use `KFold(n_splits=5, shuffle=True, random_state=0)`; Example 07 materializes those folds once and
reuses them for both response-subspace policies. Complete real-data examples are exercised by
`make examples`, not duplicated in the default test suite.

## Inspection and rendering boundary

`pipls.inspection` returns validated immutable NumPy results for Pi-PLS factor displays, shared
PLS-family latent structure, balanced biplot coordinates, observation diagnostics, and explicit-
provenance prediction diagnostics. The complete Pulp workflow keeps selection-conditioned OOF
prediction diagnostics separate from descriptive diagnostics of the final model fitted to all
development observations.

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

- package-owned block-aware scaling classes, block definitions, or block-method semantics beyond
  the implemented independent `scale_x` and `scale_y` controls;
- automatic outer validation or unbiased-performance claims;
- weighted fitting or general-purpose metadata routing;
- arbitrary nested meta-estimator support;
- public plotting helpers or style objects;
- generic dataset registries, network downloaders, or hidden preprocessing;
- publication-only analyses in this repository;
- compatibility aliases for removed pre-release APIs.

Current centering and optional scaling are not deferred: both blocks are always centered, while
`scale_x` and `scale_y` may override the backward-compatible `scale` policy independently. Learned
estimator or pipeline scaling remains fold-local during search.

## Current maintenance status

The decision lifecycle is normalized under Decision 0147. Current decisions describe durable
scientific, numerical, API, data, documentation, compatibility, and repository contracts; completed
migrations and cleanup sequences are summarized in `docs/decisions/history.md` and mapped in
`docs/decisions/retirements.md`.

Decision 0154 is now implemented in the search runtime. `PiPLSSearchCV` uses exhaustive coverage
by default over the complete fold-feasible predictor-rank domain, `max_predictor_rank=None` leaves
that domain uncapped by statistical heuristics, and `predictor_rank_values="epv"` is the explicit
fixed-rank EPV policy. The pre-release `"max"` predictor-rank policy and `"rule"` maximum-rank
sentinel are removed, and nondefault `samples_per_predictor_rank` values are valid only for EPV.
The private hard-feasibility and EPV calculations remain separate. Focused Decision-0154 regression
coverage now protects full-domain reference optima above EPV ranks, EPV component-domain resolution,
$c=1$ warning behavior, numerical-rank clipping, explicit rank domains, and adaptive full-domain
endpoints. Maintained high-dimensional examples now request adaptive coverage explicitly, while
Pulp and the synthetic entry workflows retain the exhaustive default. Generated tutorial manifests
record the active search method, exhaustive-coverage status, and effective maximum predictor rank.
The broader user and maintainer documentation now distinguishes hard feasibility, explicit rank
domain restrictions, candidate coverage, and the EPV policy. The served computational-performance
guide documents the exhaustive default cost, adaptive coverage, EPV/fixed-rank alternatives, fit
counts, validation repetitions, SVD choices, parallelism, OOF reuse, and work inspection. The
seven-patch Decision-0154 migration is complete: release notes record the breaking pre-1.0 search
change, and clean wheel/source-distribution smoke tests verify the installed full-domain exhaustive
default together with the explicit EPV policy.

Decision 0155 is accepted; Steps 1--5 are complete. The least-squares-driven response-subspace
policy remains a software extension, while the peer-reviewed cross-covariance construction remains
the default. The private core supports exactly `"cross_covariance"` and `"least_squares"`: the former
uses exact SVD of `Z.T @ Y`, while the latter uses exact reduced QR of `Z` followed by exact SVD of
`Q_Z.T @ Y`. The least-squares helper contains an explicit source comment that the construction is
not part of the peer-reviewed companion publication. `svd_solver` still governs only the predictor
decomposition.

Patches 1A--1D established Decision 0155 and its publication/API boundary. Patches 2A--2D isolated
the published response-basis calculation, added and hardened the Choice-C least-squares primitive,
and recorded the implemented private-core numerical contract. Patches 3A--3C expose
`response_subspace` on `PiPLSRegression`, validate exactly the two public values, propagate the fixed
setting through direct search, pipelines, OOF fitting, and refitting, and synchronize the implemented
public contract. Search still changes only `n_components` and `predictor_rank`; its ordinary template
continues to use `response_subspace="cross_covariance"`. No `response_subspace_` fitted provenance
attribute is introduced.

Patches 4A--4C now protect and record the stronger scientific and interoperability contract. Tests
cover an independently constructed frozen Choice-C reference, direct reduced-rank-regression
equivalence in the retained predictor coordinates, least-squares training-residual optimality, the
$q=1$ and full-response-subspace equivalence cases, and shared orthogonality/diagonalization
invariants. Public coverage includes every predictor/response scaling combination, fitted-estimator
and fitted-search pickle round trips, `set_params()` refitting, external `GridSearchCV`, and fixed
response-policy propagation. The internal mathematical and testing contracts now describe both
implemented response-subspace policies and the corrected full-domain predictor-rank search boundary.
Step 5 is complete. Patch 5A rewrites the user-facing theory to describe both response-subspace
criteria, their shared downstream factorization, the RRR interpretation of the least-squares route,
and the explicit publication/software-extension boundary. Patch 5B propagates that implemented
policy through the fixed-regression and path-selection references, reproducibility and
manuscript-alignment guidance, computational-performance trade-offs, troubleshooting, and
top-level discoverability. Patch 5C adds the maintained Pulp comparison using two searches on the
same materialized five-fold splits, reports model-development CV-MSE evidence without claiming
general superiority, and closes the user-documentation/example step. Step 6 is now active for the
stale-contract, changelog, installed-artifact, distribution, and final release audit. The
compatibility invariant remains that omitting the new parameter, or explicitly selecting
`response_subspace="cross_covariance"`, reproduces the pre-Decision-0155 fixed-estimator numerical
path subject only to ordinary floating-point behavior.

Decision 0143 owns exact selection handoff across `oof_report(selection=...)` and
`refit(selection=...)`, generic protocol-neutral OOF reporting, and fitted-model selection
provenance. Decision 0152 owns the manual selection-review presentation used by Tutorials 2 and 3.
The package has no dedicated leave-one-out mode, detector, provenance field, example, or support
promise; compatible user-supplied splitters remain ordinary interoperability.

Tests now protect behavior and machine-readable outputs rather than repository prose or source
arrangement. Distribution and documentation validation share private maintenance helpers, and the
Pulp example and tutorial renderer use caller-local plotting functions. Patch 6A adds release notes
and reconciles active maintainer/example inventory wording only; it introduces no runtime behavior.

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
