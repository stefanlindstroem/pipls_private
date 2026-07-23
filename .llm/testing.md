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
- focused benchmark runners and small example helper contracts;
- package and source-distribution contents, plus clean installed wheel and source-distribution
  runtime behavior at the public import and representative fit/predict boundary;
- presence, encoding, parsability, and structural format of shipped Markdown, YAML, CSV, TOML, and
  shell files;
- generic consistency rules, such as every indexed decision record existing and every shipped
  decision record appearing in the index;
- repository dataset tables being readable numeric comma-separated files with aligned rows;
- explicitly frozen package-benchmark results after a separate decision defines their scientific
  meaning, tolerances, and update procedure.

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

## Dataset boundary

Before benchmark fixtures are introduced, repository dataset tests should check file layout and
technical readability, not exact scientific values, row examples, column lists, dimensions, or
metadata checksums. Git history, review, public provenance, and the dataset documentation remain
the source of record for those contents.

Focused synthetic package benchmarks are executable contracts once implemented. Tests may validate
the question-specific script, deterministic generation, finite metrics, metric domains, exact
minimal CSV header, and repeatability of scientific values. Component-path API tests may verify one
ordered result per requested component count, aligned read-only arrays with stable dtypes, a numeric
predictor rank and explicit policy for every count, scalar lookup, pickling, and agreement with
conditional rows in `cv_results_`. Predictor-rank-profile tests should verify evaluated-only
ascending ranks, aligned defensive read-only arrays, scorer-general selection, invalid lookup,
pickling, and consistency with both `cv_results_` and `component_path_`. Plot and standard-PLS
helper tests should use small
synthetic inputs and verify immutable result arrays and final PDF rendering without freezing visual
pixel output.

Post-analysis inspection tests should verify mathematical identities, shapes, finite-value
validation, defensive copying, read-only results, deterministic sign handling, prediction
provenance, and absence of estimator mutation. Plot tests should use a headless backend and verify
returned figures and axes, explicit line/bar modes, label validation, and successful rendering
without pinning pixels or incidental Matplotlib artist counts. Structural tests should enforce the
single-axis package boundary and example-owned report composition.

Structural tests for Pulp, Sugarcane, and Tobacco protect direct `component_path_` access,
scikit-learn `cross_val_predict()`, immutable inspection results, explicit `ax=` calls, absence of
analytical CSV output, and the declared final PDF filenames without running the artifact-writing
scripts. Tobacco tests also protect its full-SVD configuration, source-order response pagination,
raw observation diagnostics, and caller-owned multipage PDF loops. Biplot tests protect balanced-coordinate numerics in `pipls.inspection`; maintained example and
renderer structure must expose direct Matplotlib arrows and labels, call `adjust_text()` after axis
configuration, and avoid pinning adjusted label coordinates.

Tutorial-renderer tests may run each direct calculation once in a temporary directory, parse every
declared SVG, verify manifest hashes, rank-profile metadata, and generated filenames, and enforce
Makefile/source-distribution ownership without pinning pixels or Matplotlib artist counts. Tutorial structure tests may verify the two-step navigation, generated-asset references, checked
snippet sections from examples 02 and 10, links to stable API objects, and links to stable
model-inspection anchors without freezing narrative wording. Documentation-entry tests may verify
that the README contains the two compact public workflows and tutorial routes while maintainer-only
commands and the repository map remain in `CONTRIBUTING.md`; do not pin line counts or exact
prose. Generic served-Markdown tests should resolve local files and anchors, including explicit and
mkdocstrings-generated object anchors. The API overview should keep one discoverable map of public
result objects, and troubleshooting should remain a task-oriented reference rather than a third
tutorial. Structural
tests may also require generated Pulp image references to remain tutorial-owned and keep
example-specific report implementation out of the general inspection reference. Small synthetic
matrices protect generic OOF and report contracts. One module-scoped Pulp numerical run may verify
the selected fixed pair, upper-boundary rank profile, aligned OOF predictions, and inspection shapes
without writing application artifacts.

The minimal numbered example may be protected structurally and through the package-level numerical
and plotting tests; do not duplicate its arrays as a frozen scientific fixture. Do not execute the
artifact-producing Pulp, Sugarcane, or Tobacco scripts in `make check`, and do not duplicate their
analyses as real-data benchmark scripts. `make examples` is the explicit application-validation
target and runs every numbered example, including the slower Tobacco analysis. Durable tests instead protect repository dataset
readability, the direct Pulp numerical workflow, direct Pulp, Sugarcane, and Tobacco workflow
structure, component-path API, inspection equations,
plotting contracts, and the immutable ordinary-PLS comparison-helper contract.
Do not require a universal manifest, universal schema, or shared wide result row across unrelated
benchmarks.

Package benchmark work may deliberately freeze selected shapes, values, metrics, or hashes. Such
assertions require an explicit decision record describing why the value is stable, what software
behavior it protects, and what tolerance or update procedure applies. Publication-result fixtures
belong in downstream reproduction repositories.

## Review rule

When a test reads a documentation or metadata file, ask whether the assertion protects a stable
machine contract or merely repeats today's content. Prefer a parser, executable behavior, or a
generic structural invariant. Do not use tests as a second copy of a living document.
