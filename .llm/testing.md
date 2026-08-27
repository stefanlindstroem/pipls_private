# Repository testing policy

## Purpose

Tests protect durable numerical, API, data, serialization, artifact, tool, and installed-package
behavior. They do not serve as a second copy of source structure, documentation prose, or completed
migration history.

Decision 0147 governs the behavior-focused validation boundary and the ownership of completed
integration checks.

## Behavioral boundary

A pytest assertion must observe one of these boundaries:

- a public or intentionally tested private callable result;
- estimator state, exceptions, warnings, cloning, pipelines, output containers, or serialization;
- numerical arrays, selections, reports, diagnostics, or deterministic generated data;
- a machine-readable artifact created during the test;
- a built or installed wheel or source distribution;
- an executable maintenance tool and the output it produces.

Do not open repository source, Markdown, YAML, Makefiles, manifests, workflows, or decision records
merely to search for expected literals. Do not use AST inspection to prescribe local call order,
assignments, imports, helper names, snippet markers, or plotting construction.

## Durable coverage

### Numerical core and estimator

Protect:

- fixed-model equations, dimensions, rank admissibility, and finite outputs;
- centering and scaling learned inside each fit and each training fold;
- full, randomized, and automatic predictor-SVD consistency within accepted tolerances;
- deterministic seeded behavior and subspace comparisons under sign or basis ambiguity;
- transactional fitted state after failed fits;
- copy, read-only, and overlapping-input safety;
- PLS-style methods, feature names, output containers, cloning, pipelines, and pickling;
- public warning types and their suppression boundary;
- response-subspace default compatibility and exact two-value validation;
- independent least-squares/Choice-C numerical reference behavior and reduced-rank-regression
  equivalence in the retained predictor coordinates;
- least-squares training-residual optimality for fixed $(h,r_\pi)$, together with the $q=1$ and
  full-response-subspace equivalence cases;
- shared orthogonality, ordered dilation, and regression-map factorization invariants under both
  response-subspace policies;
- least-squares behavior under all supported predictor/response scaling combinations, estimator
  pickling, `set_params()` refitting, and external scikit-learn parameter search.

### Search, selection, and OOF reporting

Protect:

- materialized split reuse, fold-local feasibility, and pipeline support;
- adaptive and exhaustive candidate behavior, exact-reference refinement, tolerance-boundary
  refinement, and the shared five-rank exhaustive-switch threshold;
- stable `cv_results_`, conditioned component paths, immutable rank profiles, and rank evidence;
- equal-split CV-MSE means and population SD for ordinary and repeated validation;
- manual, `best_score`, and dual-tolerance `minimum_cv_mse` selection;
- exact tolerance boundaries, provenance, non-mutation, refitting, and pickle behavior;
- exact selection compatibility shared by `oof_report()` and `refit()`;
- ordered OOF predictions, repeated-prediction averaging, counts, partial coverage, and pooled
  metrics;
- protocol-neutral scorer safety for realized validation-set sizes;
- absence of candidate rescoring, retained input data, or implicit final fitting in OOF reporting;
- preservation of fixed estimator `response_subspace` configuration through candidate fitting,
  pipelines, OOF reporting, search pickling, and final refitting without adding a third search
  dimension.

### Public results and inspection

Protect:

- defensive copies, read-only arrays, shape and scalar validation, direct construction, and pickle
  revalidation;
- finite decomposition, path, OOF, and inspection arrays;
- regression-map and balanced-biplot reconstruction preservation;
- observation and prediction diagnostic equations and provenance;
- current positive public exports and result fields.

Do not retain one negative tombstone test for every removed pre-release attribute or value. Use a
generic invalid-value test where accepting arbitrary invalid input remains a current behavior.

### Datasets and synthetic generation

Protect:

- dataset matrix validation, defensive read-only arrays, names, and shallow-copied top-level
  metadata;
- exact package-resource bytes, shapes, names, hashes, provenance, and licenses where byte identity
  is the accepted data contract;
- fresh loader outputs and no network or pandas dependency;
- deterministic synthetic arrays for fixed seeds;
- one active package-resource matrix pair per named reference dataset;
- loading from clean installed distributions.

### Rendering and generated artifacts

Focused renderer tests may execute a renderer with bounded synthetic or mocked numerical inputs in a
temporary directory and inspect its returned manifest and created files. Complete tutorial renderers
belong to `make docs` and are not rerun by ordinary pytest. Protect at the focused layer:

- finite and internally consistent manifest values;
- existence of every declared artifact;
- parseable generated SVG or other machine-readable output;
- deterministic semantic results across repeated rendering;
- the runtime package remaining importable without optional rendering dependencies;
- absence of a package-owned plotting module.

Do not assert exact SVG labels, colour codes, element order, marker literals, Matplotlib calls, local
helper names, or artist coordinates.

### Documentation, examples, tools, and distributions

Use the owning executable targets:

```bash
make docs
make docs-dist
make examples
make dist-check
```

These targets own strict documentation builds, source-distribution documentation, complete numbered
example execution, and clean installed-artifact behavior. The CI test workflow runs `make examples`
once on Python 3.12; the documentation workflow runs `make docs` and `make docs-dist`. Pytest may
invoke a focused maintenance tool and parse its generated output, but it must not duplicate complete
application/tutorial execution or validate these targets by reading Makefile, workflow, manifest, or
helper-script text.

Snapshot tests execute the snapshot tool and inspect the resulting archive. They may verify clean-
tree requirements, excluded generated files, and root-relative archive contents without asserting
the implementation text of the helper.

## What tests must not freeze

Do not pin:

- explanatory prose, heading wording, diagram labels, or roadmap status;
- local variable names, assignment order, imports, helper names, or snippet markers;
- example call order when executable results protect the intended behavior;
- plotting colours, marker choices, styles, label placement, or page aesthetics;
- implementation-local arrays and aliases that are not public results;
- historical spellings and removed pre-release names;
- the number of tests, files, decisions, headings, or source lines;
- performance or scientific claims not explicitly accepted by the owner.

## Test layers

### Focused tests

Run the smallest relevant behavioral modules during implementation. Focused tests support rapid
iteration but do not replace complete validation.

### Complete package checks

The authoritative local target is:

```bash
make check
```

It covers decision-registry consistency, linting, typing, and behavioral pytest coverage. Also run
`python -m compileall` and `git diff --check` when producing a patch.

### Application, documentation, and distribution checks

Run the dedicated targets when their owned behavior changes. Complete Pulp, Sugarcane, and Tobacco
workflows are application validation and must not be silently duplicated inside ordinary pytest.

### Independent patch validation

Before delivery, apply the generated patch to a fresh extraction of the same snapshot, compare every
changed file byte-for-byte with the working tree, rerun applicable checks, and verify the published
SHA-256 checksum.

Never describe a timeout, unavailable dependency, or unrun command as a pass.

## Numerical comparison policy

Numerical changes require explicit tolerances and boundary cases. Behavior-preserving refactors
require direct before-and-after comparison of affected arrays, selections, predictions, reports,
resource bytes, or manifest values. Aggregate test success alone is insufficient.

For repeated or nearly repeated singular values, compare identifiable quantities rather than raw
basis columns. Prefer semantic manifests and parsed artifact structure; retain hashes only when byte
identity itself is the contract.

## Review rule

A durable behavioral or machine-readable contract justifies a test. A statement that repeats the
current implementation, presentation, prose, or migration history belongs in review guidance, not
in an assertion.
