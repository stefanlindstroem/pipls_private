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

## Inspect-decide-refit lifecycle

Decision 0137 changes ownership rather than numerical selection. Tests protect these durable
behaviors:

- a fitted search exposes candidate evidence and immutable path/profile views;
- post-fit `refit()` accepts exactly one named rule or one component count, returns a fitted clone of
  the configured estimator or pipeline, and leaves the search unchanged;
- constructor parameters do not shadow the post-fit method, and search exposes no delegated
  prediction, transformation, scoring, inverse-transformation, or feature-name surface;
- manual component selection uses the conditionally selected predictor rank stored for that path
  row;
- `"best_score"`, `"minimum_cv_mse"`, and `"one_standard_error"` resolve through their documented
  evidence and can differ under a custom scorer;
- explicit validation reporting reuses the exact materialized search splits, preserves ordered OOF
  coverage semantics, and does not perform a full-data refit;
- the search does not retain supplied training matrices or returned fitted estimators;
- maintained search examples and tutorial renderers use `search.refit(...)` rather than manually
  reconstructing the selected fixed estimator.

The explicit report stage must additionally test single-use splitters, defensive read-only split
copies, unchanged scorer-call counts, one-dimensional response shape, repeated and partial coverage,
shape mismatch rejection, warning boundaries, and search-state immutability.

The final surface should be tested through positive constructor and method contracts plus one
compact assertion that removed pre-release constructor controls are rejected. Do not accumulate one
historical tombstone test per removed fitted attribute.

## Three-stage onboarding transition

Decision 0139 introduces staged presentation contracts rather than package behavior. Patch 2 now
protects the renamed example, generated quick-start asset, semantic manifest, navigation order, and
source-distribution documentation route. Patch 3 must complete the broader wording and reference
reframing. Durable tests protect:

- one maintained `examples/01_pulp_quick_start.py` source and no compatibility copy under the former
  name;
- checked tutorial snippets owned by that example;
- one parseable generated quick-start SVG plus a semantic manifest identifying package-owned Pulp,
  the selected rank pair, fitted-value provenance, standardized RMSE, and the SVG hash;
- strict navigation order: quick start, synthetic inspection, complete Pulp analysis;
- source-distribution inclusion and clean rendered-documentation execution;
- a single standardized observed-versus-fitted plotting axis and explicit absence of OOF or
  predictive-validation claims in the quick-start tutorial;
- synthetic-tutorial wording and structure that retain the search for evidence and manual
  component selection;
- selection-conditioned terminology for the maintained Pulp OOF workflow.

Prefer parsed navigation, executable snippets, manifest semantics, artifact existence, and focused
source-structure checks over frozen prose or pixel output. The three-page route and renamed first
example are now the tested implementation.

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

Decision 0138 adds a separate package-owned Pulp contract. Tests protect the named loader's return
modes, immutable arrays and metadata, stable names and sample identifiers, public provenance,
pickle reconstruction, resource and canonical-array hashes, clean wheel/source-distribution
loading, loader ownership by every maintained Pulp consumer, and one active Pulp matrix location.
Ordinary documentary-metadata tests should still avoid freezing narrative wording.

The `.llm/archive/pulp-repository-layout-v1/` copy is development history only and must be excluded
from runtime, distributions, served documentation, and active-dataset tests. Do not retain a
permanent tombstone assertion for the former repository path; use positive package-resource and
generic active-location checks. Sugarcane and Tobacco continue under the repository-dataset
readability contract.

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

Structural tests require maintained analytical ordinary K-fold examples and tutorial renderers to
use `KFold(n_splits=5, shuffle=True, random_state=0)` explicitly. Every rendered example that uses
an explicit splitter must also display the splitter import and definition before a shown snippet
uses its `CV` or `cv` variable; tutorial snippets must not depend on hidden module-level validation
configuration. The compact first example is the intentional exception: it uses default `cv=5` to
demonstrate the shortest installed-data workflow.
The leave-one-out example retains exhaustive `LeaveOneOut`. Structural tests for Pulp, Sugarcane,
and Tobacco protect
direct `component_path_` access, explicit `validation_report()`, immutable inspection results,
explicit Matplotlib
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

Ordinary pytest must not execute the Pulp tutorial renderer because it requires the optional
`adjustText` dependency. Static tests protect its Makefile, source-distribution, snippet, asset-name,
and ownership contracts; `make docs-figures` and `make docs` execute and validate the renderer in the
complete documentation environment. Tutorial structure tests may verify the three-step navigation,
generated-asset references, checked
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
the selected fixed pair, upper-boundary rank profile, aligned validation-report OOF predictions,
and inspection shapes
without writing application artifacts.

The Pulp quick-start example may be protected structurally and through package-level Pulp-loader,
numerical, direct-rendering, and source-distribution execution tests. Protect its chained
search/refit call, fitted-value provenance, standardized single-axis plot, and absence of OOF claims;
do not duplicate Pulp arrays as a second frozen scientific fixture. Do not execute the
artifact-producing Pulp, Sugarcane, or Tobacco scripts in `make check`, and do not duplicate their
analyses as duplicate repository scripts. `make examples` is the explicit application-validation
target and runs every numbered example, including the slower Tobacco analysis. Durable tests instead protect repository dataset readability, package-owned Pulp loading and
numerical behavior, Pulp, Sugarcane, and Tobacco workflow structure, component-path API,
inspection equations,
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
