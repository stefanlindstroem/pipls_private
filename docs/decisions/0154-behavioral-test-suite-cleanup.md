# Decision 0154: behavioral test-suite cleanup

## Status

Accepted. Patch 1 of the seven-patch cleanup sequence is complete. This decision establishes the
behavioral-testing boundary and authorizes later test deletion, maintenance-tool simplification,
caller-owned rendering refactors, and retirement of superseded implementation decisions. Runtime
code, tests, examples, and served documentation are unchanged in Patch 1.

## Context

The repository has accumulated a large structural test layer while the package and documentation
were evolving rapidly. Many tests now read Python source, Markdown, YAML, Makefiles, manifests, or
generated SVG text and search for prescribed literals. Those assertions freeze local variable
names, helper names, call order, snippet markers, headings, explanatory sentences, diagram labels,
plot colours, and removed pre-release identifiers. They often duplicate review guidance or checks
already owned by executable documentation, examples, distribution smoke tests, or the public API.

The resulting suite is harder to maintain than the behavior it protects. Small editorial or local
refactoring changes require broad test updates even when numerical results, public objects,
installed artifacts, and user-visible execution are unchanged. Some maintenance scripts also
contain duplicated environment setup, source-distribution extraction, artifact enumeration, and
embedded smoke-test code, while pytest tests those scripts by reading their implementation text.

The cleanup must not weaken the numerical or estimator contracts. It must retain tests for model
construction, search and selection, OOF reporting, immutable results, datasets, serialization,
installed distributions, generated artifacts, and executable tools. The change is about what a
test observes: behavior and machine-readable outputs rather than prescribed source or prose.

## Decision

### Test behavior, not repository wording

Pytest must not read repository files merely to assert that they contain or omit expected source or
prose literals. In particular, tests must not prescribe:

- explanatory sentences, headings, terminology scans, or roadmap wording;
- local variable, helper, region-marker, or snippet-marker names;
- source-level call order, assignment order, imports, or plotting implementation details;
- Mermaid node labels, feedback-edge text, or diagram orientation;
- Matplotlib colours, marker literals, labels, styles, or artist construction;
- Makefile, workflow, manifest, README, CONTRIBUTING, or decision-record text;
- negative tombstones for every removed pre-release name.

AST inspection is still source inspection and is not a substitute for behavioral coverage. Delete
AST-based workflow policing when the same contract can be exercised by running the code or
inspecting its public result.

### Retain machine-readable and artifact behavior

Tests may inspect files created during the test when those files are the product under test. Valid
examples include:

- parsing JSON manifests returned by a renderer;
- parsing generated SVG as XML and verifying that declared artifacts exist;
- opening built wheels and source distributions to verify installed resources;
- inspecting an archive produced by the snapshot tool;
- comparing package-resource bytes or hashes where byte identity is the dataset contract;
- parsing generated configuration produced by an executable maintenance tool.

Such tests must assert semantic structure and behavior rather than incidental serialization text.
A generated SVG may be required to parse and correspond to a manifest entry; its exact labels,
colour codes, and element ordering are not pytest contracts.

### Assign complete workflows to their owning targets

Dedicated commands own integration behavior:

- `make check` owns linting, typing, and behavioral pytest coverage;
- `make examples` owns execution of the complete numbered examples;
- `make docs` owns the strict checkout documentation build;
- `make docs-dist` owns strict source-distribution documentation builds;
- `make dist-check` owns clean wheel and source-distribution installation behavior.

Pytest should not reimplement these commands by checking the text of their Makefile targets,
workflows, manifests, or helper scripts. A small focused test may invoke a maintenance tool and
inspect its output when that is faster and isolates a meaningful behavior.

### Preserve durable runtime coverage

The cleanup must retain behavioral coverage for:

- the numerical construction, preprocessing, rank feasibility, warnings, and finite results;
- scikit-learn parameter, cloning, pipeline, output-container, feature-name, and pickle behavior;
- candidate generation, split reuse, scoring, selection, tolerances, refitting, and transactional
  fitted state;
- OOF ordering, repeated-prediction averaging, counts, partial coverage, and pooled metrics;
- immutable public results, direct construction, validation, defensive copies, and reconstruction;
- reference datasets, deterministic generators, package resources, and installed loading;
- renderer manifests, declared artifact creation, parseable outputs, and deterministic numerical
  content;
- executable snapshot, documentation, distribution, and installed-package tools.

Removed pre-release names require no permanent negative tests unless accepting the old value would
create a concrete ambiguity in the current public API. Prefer one generic invalid-value test over a
list of historical spellings.

### Simplify implementations without adding shared product surface

Maintenance scripts may share a small private module under `tools/` for subprocess, virtual
environment, archive, and artifact helpers. An installed smoke test should be a normal checked-in
script rather than a large source string embedded in another tool.

Large caller-owned rendering functions may be split into local `_plot_*` or `_render_*` helpers.
Do not add plotting utilities to `src/pipls`, a public plotting API, or a shared example plotting
framework. Numerical-core refactoring requires a separately demonstrated duplication or defect; a
long but cohesive numerical algorithm is not sufficient reason to disperse it.

### Retire completed implementation decisions after consolidation

After the test and tooling cleanup, completed implementation-sequence decisions that no longer
define an independent current contract may be summarized in `history.md`, mapped in
`retirements.md`, and removed according to Decision 0147. Durable numerical, API, dataset, and
product-boundary decisions remain current.

## Patch sequence

1. Establish this decision and synchronize the testing, development, current-state, strategy, and
   decision registries without deleting tests or changing runtime behavior -- complete.
2. Remove documentation, repository-text, decision-prose, workflow-text, and configuration-literal
   policing; retain only executable maintenance-tool behavior where justified.
3. Remove example-source, AST workflow, snippet-marker, plotting-literal, and generated-SVG-text
   policing; rewrite renderer tests around manifests, declared artifacts, and parseable outputs.
4. Consolidate duplicated distribution and documentation maintenance support, replace embedded
   installed-smoke-test source with a checked-in script, and simplify source-distribution docs
   validation to executable outcomes.
5. Remove stale tombstone and duplicate runtime tests, consolidate result and search coverage by
   ownership, and reduce exception-message coupling where wording is not public.
6. Refactor the largest caller-owned Pulp example and tutorial renderer into small local plotting
   functions without changing analytical order, numerical results, or the package surface.
7. Consolidate durable outcomes, retire superseded implementation decisions under Decision 0147,
   shorten maintainer state, and run the complete validation matrix.

## Validation obligations

The completed sequence must establish that:

- no pytest test opens source or documentation solely to search for expected literals;
- no AST test prescribes example workflow implementation or plotting construction;
- documentation and distribution correctness are exercised by their dedicated executable targets;
- renderer tests inspect semantic manifests and parseable generated artifacts;
- package-resource hashes remain only where byte identity is an accepted data contract;
- all retained numerical, estimator, selection, OOF, result, dataset, serialization, and installed
  artifact behavior remains covered;
- complete examples and strict documentation still execute successfully;
- before/after numerical arrays, selections, OOF reports, predictions, and manifest values are
  unchanged by behavior-preserving refactors;
- no plotting helper enters the installed package;
- current decisions contain durable contracts rather than completed patch narration.

The final pytest count is not a contract. A smaller suite is successful only when it removes
implementation and prose coupling while preserving meaningful behavioral coverage.

## Consequences

The suite will become smaller and less sensitive to editorial changes and local refactors. Review,
strict documentation builds, example execution, and distribution checks will own concerns that do
not belong in pytest. Maintenance tools and caller-owned renderers will have less duplicated code.

Some source-level presentation conventions will no longer fail automatically when edited. That is
intentional: those conventions remain documentation and review responsibilities unless they affect
a public result, executable artifact, or installed behavior.
