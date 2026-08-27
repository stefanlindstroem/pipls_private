# Current development state

## Purpose

This is the fresh-chat handoff for the implemented repository. It records current package behavior,
accepted exclusions, active maintenance work, and authority order. Completed migration narratives
belong in numbered decisions, `docs/decisions/history.md`, and Git history.

Read this file with `.llm/product_scope.md` before proposing work. Confirm relevant claims against
the affected source and tests in the uploaded snapshot.

## Implemented package boundary

Pi-PLS is an installable scikit-learn-compatible package for multivariate regression. The runtime
package contains:

- `PiPLSRegression`, which fits one explicit `(n_components, predictor_rank)` pair;
- `PiPLSSearchCV`, which evaluates the admissible triangular component/rank path;
- immutable component-path, selection, predictor-rank-profile, decomposition, and OOF records;
- public response-standardized MSE scorers;
- validated dataset containers with read-only model arrays, deterministic synthetic data, and
  package-owned Pulp, Sugarcane, and Tobacco reference resources;
- pure numerical fitted-model inspection under `pipls.inspection`.

The top-level wildcard surface is intentionally narrow:

```python
from pipls import PiPLSRegression, PiPLSSearchCV, PredictorRankSupportWarning
```

Result records and utility functions remain public from their focused modules. There are no
compatibility aliases for removed pre-release names.

## Modeling and selection lifecycle

`PiPLSRegression` owns fixed fitting only. It always centers both blocks and lets `scale_x` and
`scale_y` override the backward-compatible `scale` policy independently. Its `response_subspace`
parameter accepts exactly `"cross_covariance"` and `"least_squares"`; cross-covariance is the
peer-reviewed default and least-squares is an RRR-inspired software extension outside the companion
publication. Response-subspace choice is fixed-estimator configuration, not a search dimension.

`PiPLSSearchCV.fit(X, y)` materializes one validation split set, evaluates fixed-model clones, and
retains path evidence. It does not retain `X` or `y` and does not automatically fit a final
full-data model. The ordinary optimized predictor-rank domain is bounded by hard fold feasibility
and any explicit user cap. Exhaustive coverage is the default, adaptive coverage is an explicit
computational approximation, and `predictor_rank_values="epv"` is the fixed events-per-variable
policy using the full supplied sample count before fold-feasibility clipping.

Post-search work is explicit:

```python
search = PiPLSSearchCV(cv=cv).fit(X, y)
selection = search.select(rule="minimum_cv_mse")
report = search.oof_report(X, y, selection=selection)
model = search.refit(X, y, selection=selection)
```

`search.select()` is the sole public selected-row lookup and performs no fitting. The named rules are
`best_score` and tolerance-based `minimum_cv_mse`; manual selection uses an evaluated component
count and its already-conditioned predictor rank. Predictor-rank optimization applies separate
constructor-level relative and absolute configured-score tolerances before component-count
selection. A successful refit attaches the exact immutable selection as `model.selection_`.

`oof_report()` accepts an existing compatible selection and reuses every split materialized by the
search. Repeated validation predictions are averaged per observation and prediction counts are
retained. Same-search OOF diagnostics are selection-conditioned inspection, not independent
post-selection validation; independent assessment requires outer resampling or untouched external
data. Decision 0165 makes the maintained workflow boundary explicit: component-path evidence and,
when useful, the conditional predictor-rank profile complete selection before OOF diagnostics are
inspected. OOF results do not ordinarily feed back into $h$ or $r_\pi$ tuning.

## Statistical reporting

For each evaluated component-path row, validation MSE is summarized with equal weight per
materialized split:

```text
cv_mse_mean = mean(split_cv_mse)
cv_mse_std  = std(split_cv_mse, ddof=0)
```

`cv_mse_std` is descriptive split-to-split variability. Maintained CV-MSE plots use mean plus or
minus one split SD. The package exposes no standard-error result or standard-error selection rule.

For `minimum_cv_mse`, `relative_tolerance=None` resolves to
`sqrt(np.finfo(np.float64).eps)` and `absolute_tolerance=np.inf` disables the absolute cap. Both
caps must hold. Predictor-rank tolerance selection uses the analogous configured-score contract and
retains immutable evidence for optimized policies.

The default search scorer is the stable string `"neg_response_standardized_mse"`, resolving to the
public fold-local response-standardized MSE scorer. Complete real-data OOF scalar figures prefer
response-wise selection-conditioned OOF $R^2$ while keeping standardized RMSE available
numerically. Maintained response-wise $R^2$ bars cap at 1.0, do not place the lower limit above 0.0,
preserve negative values, and show the zero reference.

## Data and example boundary

Users may fit ordinary array-like `X` and `y` from any source. No generic registry, downloader,
DataFrame requirement, or package-specific ingestion layer is required. Named reference datasets
are available from `pipls.datasets` through `load_pulp()`, `load_sugarcane()`, and `load_tobacco()`;
each loader reads one canonical language-neutral package-resource matrix pair plus metadata and
licensing material.

The maintained numbered examples are user tasks:

1. compact Pulp automatic fit and fitted-value diagnostic;
2. synthetic inspect-decide-refit workflow with external-test prediction;
3. matched-fold PLS-family component-path comparison of both Pi-PLS response policies and ordinary
   PLS across the three reference datasets plus one deterministic synthetic stress case;
4. complete repeated-CV Pulp analysis;
5. complete Sugarcane analysis with EPV-fixed predictor rank at
   `samples_per_predictor_rank=5.0` and a separate component-count choice;
6. complete Tobacco analysis with optimized predictor rank and separate 20% predictor-rank and
   component-count tolerances.

Example 03 materializes one split set per comparison case and reuses it across all compared models.
Its paths are model-development evidence, not independent post-selection validation. Complete
real-data examples belong to `make examples`, while source-distribution qualification uses bounded
smoke branches where appropriate.

## Inspection and rendering boundary

`pipls.inspection` returns validated immutable NumPy results for Pi-PLS factor displays, shared
PLS-family latent structure, biplot coordinates, observation diagnostics, and explicit-provenance
prediction diagnostics. OOF diagnostics remain separate from descriptive diagnostics of the final
model fitted to all development observations.

The runtime package contains no plotting module. Examples and users own Matplotlib artists, labels,
layouts, saving, and optional annotation allocation. The maintained Pulp biplot may use optional
`textalloc>=1.2.4,<2` line-aware placement and must remain executable through a plain-Matplotlib
fallback when that dependency is absent. Exact visual tuning is not a numerical compatibility
contract.

## Documentation and distribution boundary

The repository owns the installable software product, user documentation, examples, tests,
packaging, and release validation. Paper-specific experiment grids, cached results, figure
reproduction, and publication environments belong downstream and pin a released package version.

The served site separates tutorials, programming reference, scientific background, and project
validation. Decision 0164 owns the lean flat Reference, tutorial ownership, early generated API
lookup, and the retained synthetic-generator explanation and
`docs/assets/figures/latent_geometry_generator.svg` figure. Decision 0165 refines only the
search-domain part of that architecture: the final target has separate Path and selection and OOF
diagnostics pages so that selection evidence and selection-conditioned diagnosis have distinct
owners. The Dataset API owns the synthetic generator description; publication-specific simulation
and reproduction material remains downstream rather than on a separate served page.

Patches 0164B--0164F completed the first lean-reference migration. Patches 0165A--0165D establish
the selection/OOF boundary, align the maintained tutorials, split the Reference into Path and
selection plus OOF diagnostics, and complete the terminology and cross-reference audit.
Numbered decisions and `.llm` are maintainer records and are not served as user documentation. Semantic cross-references are
maintained contextually; strict documentation builds own link resolution rather than pytest
assertions about prose placement.

Python 3.10--3.14 is supported within the dependency ranges in `pyproject.toml`. Clean wheel and
source-distribution installations are validated. `docs-dist` builds documentation from the
extracted sdist with package imports forced to that artifact's `src` tree. Snapshots are
root-relative archives of a clean committed tree.

## Explicit exclusions and deferred work

Do not add without a new owner decision:

- package-owned block-aware scaling classes or block-method semantics beyond `scale_x`/`scale_y`;
- automatic outer validation or unbiased-performance claims;
- weighted fitting or general-purpose metadata routing;
- arbitrary nested meta-estimator support;
- public plotting helpers or style objects;
- generic dataset registries, network downloaders, or hidden preprocessing;
- publication-only analyses in this repository;
- compatibility aliases for removed pre-release APIs.

## Maintenance state

Decisions 0147, 0164, and 0165 are complete. The maintained
registry contains only current decisions; completed records are summarized in
`docs/decisions/history.md` and mapped in `docs/decisions/retirements.md`. The
inherited 0153/0154 retirement-map number collisions are frozen exceptions. `make decision-check`
validates both current registries, local links, active decision references, retirement-map
uniqueness, and additional number reuse.

Tests protect executable behavior and machine-readable outputs without duplicating complete
application or tutorial workflows. `make examples` owns every numbered example and runs once in CI
on Python 3.12. `make docs` owns complete tutorial rendering and strict site integration.
Source-distribution documentation and installed artifacts remain owned by `make docs-dist` and
`make dist-check`.

## Authority and drift handling

When sources disagree, use this order:

1. an explicit owner instruction in the current request;
2. accepted numbered decisions;
3. normative `.llm` contracts, especially `product_scope.md`, `mathematics.md`,
   `numerical_contracts.md`, `public_api.md`, `data_io.md`, and `dataset_layout.md`;
4. source and tests as evidence of implemented behavior;
5. this handoff and `.llm/strategy.md`.

Do not silently choose between conflicting scientific or public-API contracts. Resolve the exact
conflict in the same patch or obtain an owner decision.

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
