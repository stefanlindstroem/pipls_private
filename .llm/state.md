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
3. matched-fold PLS-family component-path comparison across Pulp, Sugarcane, Tobacco, and one
   deterministic near-saturated synthetic stress case, covering both Pi-PLS response-subspace
   policies and ordinary PLS without final refitting;
4. complete repeated-CV Pulp analysis;
5. complete Sugarcane analysis;
6. complete Tobacco analysis with separate 10% predictor-rank and component-count tolerances.

Example 04 uses `RepeatedKFold(n_splits=5, n_repeats=10, random_state=0)`. Examples 03, 05, and 06
use `KFold(n_splits=5, shuffle=True, random_state=0)`. Example 03 materializes those folds once per
comparison case and reuses the exact same split object for the cross-covariance Pi-PLS search,
least-squares Pi-PLS search, and ordinary-PLS path. Complete real-data examples are exercised by
`make examples`, not duplicated in the default test suite.

Decisions 0156 and 0157 are implemented and closed. Example 03 is the sole maintained PLS-family
comparison and overlays the cross-covariance Pi-PLS path, least-squares Pi-PLS path, and ordinary
PLS path for Pulp, Sugarcane, Tobacco, and one deterministic near-saturated synthetic stress case.
Every case uses one materialized five-fold protocol shared across all three methods. The synthetic
case fixes 25 observations, 40 predictors, 10 responses, 5 shared directions, 15 predictor-specific
directions, no response-specific directions, common noise SD 0.3, and `random_state=0`; both Pi-PLS
policies use exhaustive predictor-rank coverage over components 1 through 10. In this fixed
realization, the minimum mean CV-MSE occurs at 5 components for both Pi-PLS policies (0.9036
cross-covariance; 0.9022 least squares), while ordinary PLS reaches 0.9373 at 8 components. This is
exploratory model-development evidence and does not establish a general performance ordering.
Source-distribution qualification executes only the bounded Pulp branch of Example 03; `make
examples` owns complete four-case execution. The former Example 07 and its dedicated PDF remain
retired.

Decision 0158 is implemented and closed. Example 03 and the dedicated Home renderer share
`examples/_support/pls_family_path_comparison.py`, so the simplified Pulp/Tobacco Home figures use
the same seeded materialized five-fold protocol and case-specific Pi-PLS search settings as the
maintained comparison. The Home renderer requests only the publication-default
`"cross_covariance"` policy plus ordinary PLS, writes concise `Π-PLS` versus `PLS` SVGs and a
semantic manifest under `docs/assets/generated/home/`, and is part of `docs-figures` and the source
distribution. The two figures are placed side by side under `Why use Π-PLS?`; the accompanying text
limits the parsimony interpretation to shared component count $h$, notes that Pi-PLS also selects
$r_\pi$, and links to Example 03 for the complete matched-validation comparison.

Decision 0159 is implemented and closed. Patch 0159A establishes semantic documentation
cross-referencing and explicit canonical anchors for the three reference-dataset detail sections,
the companion-paper citation section, and frequently referenced theory concepts. Patch 0159B adds
semantic Pulp, Sugarcane, and Tobacco links across the served documentation, including direct
same-page dataset-guide navigation, while preserving tutorial, workflow, loader, heading, code, and
figure-alt semantics. Patch 0159C adds canonical companion-publication links and concept-specific
theory links across Home, API, performance, inspection, path-analysis, reproducibility,
troubleshooting, synthetic-data, and example guidance. Patch 0159D completes the navigation audit,
connects compatibility to reproducibility, performance guidance to path-selection semantics, and the
dataset guide to the dataset API, and adds `tests/test_documentation_cross_references.py` to protect
canonical anchors, representative semantic routes, and against isolated served pages. The policy
favors one meaningful destination per reference over mechanical link density.

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

Decision 0155 is implemented and closed. `PiPLSRegression.response_subspace` accepts exactly
`"cross_covariance"` and `"least_squares"`; the peer-reviewed cross-covariance construction remains
the default, while the least-squares/RRR-inspired policy is a software extension outside the
companion publication. The core keeps response-side algebra exact: cross-covariance uses exact SVD
of `Z.T @ Y`, least-squares uses exact reduced QR of `Z` followed by exact SVD of `Q_Z.T @ Y`, and
the final SVD of `W` is exact under both policies. `svd_solver` continues to govern only the
predictor decomposition.

`response_subspace` is fixed-estimator configuration, not a third `PiPLSSearchCV` search dimension.
Candidate evaluation, OOF work, pipelines, serialization, external `GridSearchCV`, and final
refitting preserve the estimator-template policy while package search changes only `n_components`
and `predictor_rank`. The ordinary search template remains cross-covariance, and no
`response_subspace_` fitted provenance attribute is introduced. Maintained theory, API,
reproducibility, performance, troubleshooting, manuscript-alignment, and Example-03 material keep
the publication/software-extension boundary explicit. The compatibility invariant is that omitting
the parameter, or explicitly selecting `response_subspace="cross_covariance"`, preserves the
pre-Decision-0155 fixed-estimator numerical path subject only to ordinary floating-point behavior.

Decision 0143 owns exact selection handoff across `oof_report(selection=...)` and
`refit(selection=...)`, generic protocol-neutral OOF reporting, and fitted-model selection
provenance. Decision 0152 owns the manual selection-review presentation used by Tutorials 2 and 3.
The package has no dedicated leave-one-out mode, detector, provenance field, example, or support
promise; compatible user-supplied splitters remain ordinary interoperability.

Tests protect behavior and machine-readable outputs rather than repository prose or source
arrangement. Distribution and documentation validation share private maintenance helpers, and the
Pulp example and tutorial renderer use caller-local plotting functions. Clean wheel and
source-distribution qualification exercises both response-subspace policies, verifies least-squares
search/refit propagation, and runs Example 01 plus the bounded Pulp branch of Example 03 from the
extracted sdist.

`docs-dist` separately verifies that documentation can be built from the extracted source
distribution. It uses the invoking maintained documentation environment but forces `PYTHONPATH` to
the extracted artifact's `src` tree, so package code and documentation content cannot fall back to
the development checkout. Clean installation isolation remains the responsibility of `dist-check`;
`docs-dist` therefore does not reinstall the scientific and documentation dependency stack into a
second temporary virtual environment. Artifact installation checks reuse pip's normal cache,
including a caller-supplied `PIP_CACHE_DIR`.

Decision 0160 is implemented and closed. `adjustText` remains installed by the maintained
`examples`, `docs`, and `dev` extras but is no longer required for successful Pulp rendering.
Example 04 and the Pulp tutorial renderer retain their original Matplotlib text positions when
`adjustText` itself is unavailable, while unrelated import failures from an installed
`adjustText` continue to propagate. The runtime package remains independent of both graphics
dependencies.

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
