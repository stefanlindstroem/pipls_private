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
- executable package examples and benchmark runners;
- package and source-distribution contents;
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
file exists and parses as a non-empty YAML mapping rather than pinning its contents.

## Dataset boundary

Before benchmark fixtures are introduced, repository dataset tests should check file layout and
technical readability, not exact scientific values, row examples, column lists, dimensions, or
metadata checksums. Git history, review, public provenance, and the dataset documentation remain
the source of record for those contents.

Package benchmark manifests and result schemas are executable, versioned contracts rather than
living documentary metadata. Tests may therefore validate their required fields, references, rank
constraints, metric domains, schema structure, CSV header order, and typed round-trip behavior when
an accepted benchmark decision defines those semantics.

Package benchmark work may deliberately freeze selected shapes, values, metrics, or hashes. Such
assertions require an explicit decision record describing why the value is stable, what software
behavior it protects, and what tolerance or update procedure applies. Publication-result fixtures
belong in downstream reproduction repositories.

## Review rule

When a test reads a documentation or metadata file, ask whether the assertion protects a stable
machine contract or merely repeats today's content. Prefer a parser, executable behavior, or a
generic structural invariant. Do not use tests as a second copy of a living document.
