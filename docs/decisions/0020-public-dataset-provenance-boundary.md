# Decision 0020: public dataset provenance boundary

## Status

Accepted during Phase E3. The Corn-specific reconstruction plan was superseded by Decision 0041.

## Context

Repository dataset materials are visible to programming users. References to private delivery
archives, inaccessible local source paths, personal handoffs, or preparation-only scripts can make
public assets confusing and irreproducible even when the committed model matrices are complete.

## Decision

- Dataset-facing provenance cites public sources or raw files included in the repository.
- Private archive names, personal delivery details, inaccessible paths, and checksums for unshared
  inputs are not committed as public dataset metadata.
- Preparation-only scripts that require unavailable development inputs remain outside the public
  repository.
- Simple adaptations are described directly in the dataset metadata and README.
- Public reconstruction code is included only when it operates on included or publicly obtainable
  raw data and exposes analysis-relevant choices that programming users should see.
- At the time of this decision, Corn was designated as a possible public reconstruction case.
  Decision 0041 later excluded it because source-level redistribution rights were not sufficiently
  explicit and its package-level value did not justify the uncertainty.

## Consequences

- Every committed dataset can be understood from public repository materials alone.
- Programming users are not directed toward unavailable files or internal project tooling.
- Internal development lineage does not become part of the runtime or example contract.
- Any future public reconstruction remains transparent and is admitted only after the exact
  source material passes the licensing gate established by Decision 0041.
