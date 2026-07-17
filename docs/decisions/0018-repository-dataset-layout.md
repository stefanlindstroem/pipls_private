# Decision 0018: repository real-dataset layout

## Status

Accepted during Phase E3; clarified by Decision 0020.

## Context

Transparent examples should resemble ordinary user code, but committed datasets also need a
consistent, reviewable repository layout. The first Linnerud integration used source-specific file
names and whitespace-delimited tables, which would make later dataset directories unnecessarily
heterogeneous.

## Decision

- Every committed real dataset uses `X.csv`, `Y.csv`, and `metadata.yaml`.
- Both model tables are UTF-8, comma-delimited CSV with headers.
- `metadata.yaml` follows the versioned repository contract in `.llm/dataset_layout.md` and records
  descriptions, public source, licensing, dimensions, variables, alignment, missingness,
  preparation, and integrity hashes.
- Metadata is required for repository inclusion but is not read by the estimator, examples, or
  external programming users.
- Examples continue to read `X.csv` and `Y.csv` directly and visibly.
- Dataset assets must not expose private archive names, inaccessible local paths, or internal
  preparation scripts.
- Public reconstruction code is exceptional and must operate on included or publicly obtainable
  raw data while exposing analysis-relevant choices.
- No public registry, generic loader, automatic download, or metadata-driven runtime path is added.

## Consequences

- Dataset directories are predictable and comparable across future integrations.
- Examples remain transparent and representative of the intended programming-user workflow.
- Repository reproducibility is strengthened without broadening the public API or leaking internal
  development materials.
- Existing real-dataset integrations must be normalized to this layout and provenance boundary.
