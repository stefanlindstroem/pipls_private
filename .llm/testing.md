# Repository testing policy

## Purpose

Tests protect durable numerical, API, repository, documentation, data, and distribution contracts.
They must not become a second copy of living prose or preserve removed migration scaffolding.

## What tests should protect

### Numerical core and estimator

- fixed-model equations, dimensions, rank admissibility, and finite outputs;
- centering/scaling learned inside each fit and each training fold;
- full/randomized/auto predictor-SVD consistency within stated tolerances;
- deterministic behavior for fixed seeds and stable subspace comparisons under sign or basis
  ambiguity;
- transactional fitted state after failed fits;
- copy/read-only/overlapping-input safety;
- PLS-style methods, feature names, output containers, cloning, and pickling;
- exact public warning type and suppression boundary.

### Search and selection

- materialized split reuse and fold-local feasibility;
- exhaustive and adaptive candidate behavior, private numerical score ties, tolerance-independent
  candidate coverage, and pipeline support;
- stable `cv_results_`, conditioned component path, exact and retained rank-profile selections, and
  immutable predictor-rank evidence;
- equal-split CV-MSE means and `ddof=0` SD, including repeated CV, one split, and unequal validation
  lengths;
- conditioned-path `best_score`, manual component selection, and dual-tolerance `minimum_cv_mse`
  behavior;
- independent predictor-rank and component-count tolerance defaults, simultaneous caps, exact
  boundary inclusion, positive, negative, and zero score references, invalid arguments, provenance,
  search non-mutation, refit, pipelines, and pickle behavior;
- absence of retired standard-error properties and rules.

### OOF reporting

- exact compatibility with the supplied selection and fitted search;
- reuse of every stored split without rematerialization;
- ordered predictions, repeated-prediction averaging, counts, and partial coverage;
- pooled OOF R2 over covered rows only;
- protocol-neutral scorer safety for realized validation-set sizes;
- no candidate rescoring, search mutation, retained input data, or implicit full-data fit.

Generic OOF ordering, averaging, counts, partial coverage, pooled OOF R2, and singleton-validation
scorer safety are the durable contracts. API tests use ordinary $K$-fold or explicit split
iterables and do not establish protocol-specific package support.

### Immutable public results and inspection

- defensive copies, read-only arrays, shape/scalar validation, direct construction, and pickle
  revalidation;
- finite derived decomposition, path, OOF, and inspection arrays;
- display-factor regression-map preservation;
- balanced biplot reconstruction preservation;
- observation-diagnostic equations;
- prediction provenance, residual orientation, standardization, and RMSE;
- absence of public plotting helpers and private construction state.

### Datasets and synthetic generation

- immutable `PiPLSDataset`, recursively frozen metadata, names, sample identity, and truth records;
- exact Pulp, Sugarcane, and Tobacco resource bytes, shapes, names, hashes, provenance, and
  licenses;
- fresh loader outputs, read-only `(X, Y)` returns, and no network/pandas dependency;
- deterministic synthetic arrays and truth for fixed seeds;
- one active package-resource matrix pair per named reference dataset;
- clean wheel and source-distribution inclusion and loading.

### Examples, rendering, and documentation

- public imports, visible data preparation, seeded CV, route-specific
  unselected-path/selection/review/refit order, feedback-aware tutorial structure, and
  caller-owned rendering;
- caller-owned Matplotlib rendering from immutable arrays;
- no private runtime imports or duplicate workflow implementations;
- quick-start executable behavior and numbered-example structural contracts;
- focused module-scoped Pulp numerical coverage where justified;
- maintained tutorial snippets, semantic manifests, asset names, parseable SVGs, and stable links;
- strict MkDocs build, generated API ownership, local links/anchors, source-distribution docs, and
  Pages workflow boundaries;
- computational-performance examples with valid public constructor keywords, current search and
  solver values, seeded randomized operations, one-candidate policy clarity, and distinction
  between fit counts and wall time;
- final generated examples through `make examples`, not ordinary `make check`.

### Repository and release policy

- required tracked files and UTF-8 Markdown structure;
- public-module `__all__` boundaries;
- package metadata, supported Python/dependency ranges, citation, authorship, and licenses;
- clean wheel and source-distribution smoke tests;
- clean root-relative snapshot and patch helper behavior;
- current-decision index integrity, historical-summary anchors, exhaustive retirement mapping, and
  active-reference integrity under Decision 0147.

## What tests should not freeze

Do not pin:

- exact explanatory prose, line counts, heading wording beyond stable navigation contracts, or
  roadmap status paragraphs;
- local variable names, incidental assignment patterns, or helper names when executable behavior
  already protects the contract;
- Matplotlib styles, artist coordinates, optional label-adjustment positions, or page aesthetics;
- implementation-local arrays and aliases that are not public results;
- one tombstone assertion for every removed pre-release name;
- a fixed final number of decision records;
- performance or scientific claims not explicitly accepted by the owner.

When source-structure testing is necessary, prefer reusable AST predicates and parameterization over
large repeated source-text checks.

## Test layers

### Focused tests

Run the smallest relevant modules during implementation. A focused command is evidence for rapid
iteration, not a substitute for complete validation.

### Complete package checks

The authoritative local target is:

```bash
make check
```

It should cover formatting/linting, typing, unit/API/repository tests, and any configured package
checks. Also run `python -m compileall` and `git diff --check` when producing a patch.
Documentation,
distribution, or example changes require their dedicated targets.

### Application and documentation checks

Use as applicable:

```bash
make docs
make docs-figures
make dist-check
make examples
```

Complete Pulp, Sugarcane, and Tobacco workflows are application validation and may be expensive.
Do not silently duplicate them inside ordinary pytest.

### Independent patch validation

Before delivery, apply the generated patch to a fresh extraction of the same snapshot, compare every
changed file byte-for-byte with the working tree, and rerun all applicable checks. Verify the
published SHA-256 checksum.

If one monolithic pytest process is unreliable in the execution environment, run a complete set of
nonoverlapping test groups and report the grouping honestly. Never describe a timeout or unavailable
tool as a pass.

## Numerical comparison policy

Numerical changes require explicit tolerances and boundary cases. Behavior-preserving refactors
require direct before/after comparisons of the affected arrays, selections, predictions, reports,
resource bytes, or generator outputs. Do not rely only on aggregate test success.

For repeated or nearly repeated singular values, compare identifiable quantities rather than raw
basis columns. For deterministic assets, prefer semantic manifests and parsed structure over only a
binary hash; retain hashes when byte identity itself is the contract.

## Documentation test policy

When a test reads Markdown or metadata, ask whether it protects a stable machine contract. Prefer:

- parsable configuration;
- executable snippets;
- generic local-link and anchor resolution;
- required navigation destinations;
- public object availability;
- source-distribution buildability.

Avoid exact prose assertions. Historical decisions and changelog entries may retain terminology
that is intentionally absent from the active API.

## Predictor-rank tolerance obligations

Tests verify that:

- private exact-score comparison remains distinct from public tolerance qualification;
- changing only predictor-rank tolerances does not change evaluated candidates, split scores,
  candidate summaries, `rank_test_score`, or adaptive/exhaustive diagnostics;
- `"adaptive"` selects only among evaluated ranks and `"exhaustive"` selects among all admissible
  ranks, with byte-identical evidence under the Decision 0149 mapping from the former values;
- fixed and maximum policies reject nondefault tolerances and expose no inapplicable evidence;
- optimized path rows, selections, profiles, component references, OOF reports, clones, pipelines,
  and pickles retain complete validated predictor-rank provenance;
- Tobacco applies and labels separate 10% predictor-rank and component-count tolerances.

## Predictor-rank search terminology obligations

Tests verify that:

- `"adaptive"` is the constructor default and preserves deterministic coarse-to-fine coverage;
- `"exhaustive"` evaluates every admissible predictor-rank candidate;
- the former pre-release values are rejected without aliases;
- cloning, parameter surfaces, repr, pipelines, and pickles contain only the current values;
- `search_is_exhaustive_` remains an achieved-coverage diagnostic and can be true after an adaptive
  request;
- maximum and one-element fixed-rank policies accept the default and reject nondefault exhaustive
  coverage;
- multi-rank explicit sequences retain adaptive and exhaustive choices;
- current source, examples, public guides, and retained decisions use the final terminology while
  unrelated `svd_solver="auto"` uses remain intact.

## Selection-driven refit and tutorial-workflow obligations

Tests must verify that:

- exactly one of `selection`, `rule`, and `n_components` configures `refit()`;
- `refit(selection=...)` and `oof_report(selection=...)` share one exact compatibility contract;
- nondefault component-count tolerances are rejected when a selection is supplied;
- compatible direct-estimator and terminal-pipeline refits attach the exact supplied immutable
  selection only after successful fitting;
- invalid types, incompatible provenance, and failed fits do not mutate the search or leave partial
  selection state;
- rule-based and manual component-count refitting remain supported;
- manual evidence-retaining examples inspect an unselected component path before defining the
  chosen component count, create one selection, then inspect the selected path, conditional rank,
  and OOF evidence before final refitting, while validation-only and comparison-only examples do
  not acquire unnecessary final models;
- affected tutorial diagrams contain one feedback edge from selected-evidence review to selection,
  initial path artifacts omit a selected marker, and selected path artifacts contain it;
- before/after selected pairs, OOF arrays, fitted predictions, and generated numerical results are
  unchanged;
- each served tutorial has one vertical Mermaid flowchart, equivalent prose, and strict local,
  Pages-overlay, and source-distribution rendering without committed diagram assets;
- active public workflow guides describe analytical routes with a pre-refit `search.select()`
  handoff, do not recover the working selection from `model.selection_`, and do not call same-search
  OOF reporting independent qualification or validation.

## Computational-performance documentation obligations

Tests should protect the guide as a stable documentation contract rather than freeze its prose.
Verify its Reference navigation position, section structure, current public parameter names, seeded
randomized examples, omission of `search_method` for one-candidate rank policies, fold-local
preprocessing boundary, and distinction between candidate-fit counts and wall-clock timing. Final
distribution checks must include the served page.

## Decision lifecycle and repository-hygiene obligations

Tests must verify:

- every shipped numbered decision is indexed;
- every retained current decision, history link, and retirement replacement resolves;
- retirement maps are nonconflicting and retired numbers are never reused;
- active maintainer and decision records do not reference retired records;
- no test pins the retained decision count or copies deleted decisions into tombstone fixtures;
- `history.md` summarizes outcomes rather than reproducing retired files;
- ignored untracked caches and generated files remain absent from snapshots, while equivalent
  deliberately tracked artifacts cause snapshot creation to fail clearly;
- the public `pipls.datasets` façade, module identity, resources, deterministic generators,
  immutability, pickling, and installed-distribution behavior remain stable across internal splits.

## Review rule

A durable behavioral or machine-readable contract justifies a test. A statement that merely repeats
today's implementation or prose should remain review guidance, not an assertion.
