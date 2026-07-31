# Repository testing policy

## Purpose

Tests should protect executable behavior and durable repository contracts without turning living
planning documents or documentary metadata into frozen implementation data. A documentation edit
must not require changing tests merely because wording, roadmap status, examples, citations, or
descriptive metadata values changed.

## What tests should protect

Tests may verify:

- public Python behavior, numerical invariants, validation, fitted attributes, and supported
  scikit-learn composition;
- small example helper contracts;
- package and source-distribution contents, plus clean installed wheel and source-distribution
  runtime behavior at the public import and representative fit/predict boundary; extracted source
  distributions must retain example output-directory placeholders and run example 01 after the
  `examples` extra is installed;
- optional dependency metadata matching maintained `dev`, `examples`, and `docs` workflows, with
  public source-install commands remaining noneditable and contributor setup remaining editable;
- presence, encoding, parsability, and structural format of shipped Markdown, YAML, CSV, TOML, and
  shell files;
- generic consistency rules, such as every indexed decision record existing and every shipped
  decision record appearing in the index, and every top-level runtime Python module appearing in the
  project ownership map;
- snapshot integrity in isolated Git repositories: clean committed-tree contents, refusal of
  tracked/staged/nonignored-untracked changes, exclusion of ignored files, truthful metadata, and
  refusal of committed generated files below `examples/results/`;
- the Git-tracked example-result tree containing only `.gitkeep` directory placeholders, while
  ignored local outputs remain permitted;
- repository dataset tables being readable numeric comma-separated files with aligned rows;

## What tests should not freeze

Tests must not assert current prose fragments or individual documentary field values in:

- `.llm/state.md`, `.llm/strategy.md`, roadmaps, handoff text, or maintenance instructions;
- changelogs, README prose, narrative documentation, or example commentary;
- dataset titles, summaries, descriptions, citations, provenance wording, preparation narrative,
  dimensions, variable lists, or integrity values stored in `metadata.yaml`;
- a hard-coded list of historical decision filenames when index-to-directory consistency can be
  checked generically.

The repository may still document required metadata sections in `.llm/dataset_layout.md`. Review
and schema evolution govern that documentary contract; ordinary tests check that each metadata
file exists and parses as a non-empty YAML mapping rather than pinning its contents. A generic
consistency test may collect DOI values from each metadata source block and require matching
resolvable links in the served dataset guide; it must not hard-code the current DOI values.

## Historical removal boundary

Do not keep a test solely to prove that a former internal name, helper, dataset integration,
documentation wrapper, or repository path remains absent. Accepted decision records preserve that
history. Prefer positive tests of the supported surface and generic structural checks.

A negative test remains appropriate when absence is itself a current public or architectural
contract, such as the package exposing no plotting API, optional rendering libraries remaining
outside runtime dependencies, or repository datasets remaining outside top-level exports. Do not
turn every pre-release deletion into a permanent executable tombstone.

## Rendering validation boundary

Protect rendering through executable artifact generation, parseable declared outputs, optional
dependency boundaries, caller-owned chart construction, direct immutable-result use, and numerical
semantics such as fold-based standard errors. Same-file private rendering functions remain
caller-owned; structural tests may verify that scientific computation stays in `main()` and that
those functions do not load data, fit models, run cross-validation, select models, or calculate
inspection results. Do not freeze exact private function names or signatures, title text, axis-label
wording, Matplotlib call counts, source-code ranges, axis-limit expressions, tick-label expressions,
or label rotation syntax. Review regenerated SVG and PDF artifacts when visible rendering behavior
changes.

## Dataset boundary

Repository dataset tests should check file layout and
technical readability, not exact scientific values, row examples, column lists, dimensions, or
metadata checksums. Public container tests should verify recursive freezing, defensive array copies,
acceptance of non-object metadata arrays, and rejection of object-dtype arrays whose elements could
remain mutable. Git history, review, public provenance, and the dataset documentation remain the
source of record for documentary contents.

Post-analysis inspection tests should verify mathematical identities, shapes, direct-construction
and pickle invariants, finite-value validation, defensive copying, read-only results, deterministic
predictor- and response-anchored sign handling, zero-anchor fallback, prediction provenance,
representable extreme inputs, explicit failure for
unrepresentable derived values, and absence of estimator mutation. Rendering tests belong at the example or tutorial
boundary and should use a headless Matplotlib backend. Protect named result-field access, direct
figure and axis construction, physical coordinate order, final files, and successful SVG or PDF
creation without pinning pixels, exact styling, automatically adjusted label positions, or
incidental artist counts. Structural tests should require the absence of a package plotting module
and public `plot_*` functions, keep Matplotlib and `adjustText` optional, and ensure that example
support code does not hide chart construction.

Structural tests for Pulp, Sugarcane, and Tobacco protect direct `component_path_` access,
scikit-learn `cross_val_predict()`, immutable inspection results, explicit Matplotlib
construction, absence of analytical CSV output, physical coordinate order, and the declared final
PDF filenames without running the artifact-writing scripts. Keep these source scans consolidated as
AST-level ownership and data-flow checks rather than repeating per-example string inventories.
Sugarcane and Tobacco tests also protect the boundary between `main()`-owned analysis and private
same-file rendering. Tobacco tests retain its full-SVD configuration, source-order response
pagination, raw observation diagnostics, and two same-file multipage report loops. Biplot tests
protect
balanced-coordinate numerics in `pipls.inspection`; maintained example and
renderer structure must expose direct Matplotlib arrows and labels, call `adjust_text()` after axis
configuration, and avoid pinning adjusted label coordinates.

Tutorial-renderer tests may run each direct calculation once in a temporary directory, parse every
declared SVG, verify manifest hashes, rank-profile metadata, and generated filenames, and enforce
Makefile/source-distribution ownership without pinning pixels or Matplotlib artist counts. Tutorial structure tests may verify the two-step navigation, generated-asset references, checked
snippet sections from examples 02 and 05, links to stable API objects, and links to stable
model-inspection anchors without freezing narrative wording. Documentation-entry tests may verify
that the README contains the two compact public workflows and tutorial routes while maintainer-only
commands and the repository map remain in `CONTRIBUTING.md`; do not pin line counts or exact
prose. Maintained user-facing examples and tutorial renderers should demonstrate current public
defaults rather than repeat redundant default arguments; tests may prohibit stale explicit defaults
while leaving historical decision records unchanged. Generic served-Markdown tests should resolve
local files and anchors, including explicit and
mkdocstrings-generated object anchors. The API overview should keep one discoverable map of public
result objects, and troubleshooting should remain a task-oriented reference rather than a third
tutorial. Generated API pages should suppress constructor signatures for returned immutable result
records while keeping the directly constructible `PiPLSDataset` signature visible. Tests should
protect the absence of sign-canonicalization bookkeeping and structurally impossible zero loading
blocks without pinning private local calculations. Structural
tests may also require generated Pulp image references to remain tutorial-owned and keep
example-specific report implementation out of the general inspection reference. Small synthetic
matrices protect generic OOF and report contracts. The focused leave-one-out example may be executed
directly because it is small and artifact-free; tests should check labeled output and semantic
coverage without freezing exact selected scores. One module-scoped Pulp numerical run may verify
the selected fixed pair, upper-boundary rank profile, aligned OOF predictions, and inspection shapes
without writing application artifacts.

The minimal numbered example may be protected structurally and through the package-level numerical
and direct-rendering tests; do not duplicate its arrays as a frozen scientific fixture. Do not execute the
artifact-producing Pulp, Sugarcane, or Tobacco scripts in `make check`, and do not duplicate their
analyses as duplicate repository scripts. `make examples` is the explicit application-validation
target and runs every numbered example, including the slower Tobacco analysis. Durable tests instead protect repository dataset
readability, the direct Pulp numerical workflow, direct Pulp, Sugarcane, and Tobacco workflow
structure, component-path API, inspection equations,
direct rendering from immutable inspection arrays and the immutable ordinary-PLS comparison-helper contract.
Do not require a universal manifest, universal schema, or shared wide result row across unrelated
comparative studies.

Documentation workflow tests should protect strict checkout and source-distribution validation,
master-only Pages deployment, least-privilege deployment permissions, canonical repository-derived
configuration, and generated-artifact upload. They should not pin action implementation details
beyond the maintained official Pages action majors.


## Review rule

When a test reads a documentation or metadata file, ask whether the assertion protects a stable
machine contract or merely repeats today's content. Prefer a parser, executable behavior, or a
generic structural invariant. Do not use tests as a second copy of a living document.
