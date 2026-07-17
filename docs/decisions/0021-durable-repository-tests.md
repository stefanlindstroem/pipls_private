# Decision 0021: durable repository tests

## Status

Accepted after the Phase E3 repository tests coupled a living strategy document to fixed prose and
an enumerated decision filename.

## Context

The repository contains living handoff, roadmap, documentation, and dataset metadata files. Tests
that copy their current prose or individual field values become stale whenever those documents are
properly updated. This creates false failures and encourages synchronized test edits that do not
protect executable behavior.

## Decision

- Tests protect executable behavior and stable machine-facing contracts.
- Shipped documentation and metadata may be tested for existence, encoding, parsability, and
  structural format.
- Tests do not pin roadmap prose, current phase wording, narrative documentation, descriptive
  metadata values, exact dataset fields, or hard-coded historical decision lists.
- Cross-file consistency is tested generically where possible, such as comparing the decision index
  with the decision-record directory.
- Exact dataset values and benchmark metrics are frozen only through an explicit benchmark or
  reproduction decision with documented tolerances and update rules.

## Consequences

- Living documents can evolve without unrelated test maintenance.
- Missing or malformed shipped files remain detectable.
- Behavioral regressions continue to be tested at the public or numerical boundary.
- Future Phase E4 fixtures can deliberately introduce stable scientific expectations rather than
  inheriting accidental assertions from repository-document tests.
