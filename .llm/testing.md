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
- exhaustive and adaptive candidate behavior, score tolerance, tie breaking, and pipeline support;
- stable `cv_results_`, component path, and conditional rank-profile evidence;
- equal-split CV-MSE means and `ddof=0` SD, including repeated CV, one split, and unequal validation
  lengths;
- `best_score`, manual component selection, and dual-tolerance `minimum_cv_mse` behavior;
- default tolerance resolution, simultaneous caps, exact boundary inclusion, zero minimum, invalid
  arguments, provenance, search non-mutation, refit, pipelines, and pickle behavior;
- absence of retired standard-error properties and rules.

### OOF reporting

- exact compatibility with the supplied selection and fitted search;
- reuse of every stored split without rematerialization;
- ordered predictions, repeated-prediction averaging, counts, partial coverage, and leave-one-out
  provenance;
- pooled OOF R2 over covered rows only;
- no candidate rescoring, search mutation, retained input data, or implicit full-data fit.

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

- public imports, visible data preparation, seeded CV, modeling-before-analysis, and rendering-last
  structure;
- caller-owned Matplotlib rendering from immutable arrays;
- no private runtime imports or duplicate workflow implementations;
- quick-start and small leave-one-out executable behavior;
- focused module-scoped Pulp numerical coverage where justified;
- maintained tutorial snippets, semantic manifests, asset names, parseable SVGs, and stable links;
- strict MkDocs build, generated API ownership, local links/anchors, source-distribution docs, and
  Pages workflow boundaries;
- final generated examples through `make examples`, not ordinary `make check`.

### Repository and release policy

- required tracked files and UTF-8 Markdown structure;
- public-module `__all__` boundaries;
- package metadata, supported Python/dependency ranges, citation, authorship, and licenses;
- clean wheel and source-distribution smoke tests;
- clean root-relative snapshot and patch helper behavior;
- decision-index and active-link integrity under Decision 0147.

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

## Decision 0147 obligations

During decision consolidation, tests must verify:

- every shipped numbered decision is indexed until retirement;
- every retained current decision and historical-summary link resolves;
- retirement maps cover every deleted record and decision numbers are never reused;
- active `.llm`, tests, documentation, and retained decisions have no broken references;
- no test pins the final retained decision count;
- `history.md` summarizes outcomes rather than copying retired files.

The snapshot-hardening patch must demonstrate both sides of the policy: ignored untracked caches and
generated files remain absent from snapshots, while deliberately tracked cache, bytecode, coverage,
built-site, build, or generated-example artifacts cause snapshot creation to fail clearly.

The dataset-module split must preserve public imports, resources, generated arrays, validation,
immutability, pickling, and distribution contents exactly.

## Review rule

A durable behavioral or machine-readable contract justifies a test. A statement that merely repeats
today's implementation or prose should remain review guidance, not an assertion.
