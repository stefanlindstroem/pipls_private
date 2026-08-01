# Decision 0136: seeded shuffled folds in maintained examples

## Status

Accepted and implemented.

## Context

The maintained examples previously relied on ordered or non-shuffled five-fold partitions. That
made the fold assignment depend directly on repository row order. The behavior was deterministic,
but the examples could inadvertently preserve ordering artifacts that have no modeling meaning.

The package default `cv=5` remains a standard scikit-learn regression-CV default. The question here
is narrower: which explicit partition should the maintained examples use when demonstrating
ordinary five-fold model selection, path comparison, and selection-conditioned out-of-fold (OOF)
prediction.

## Decision

Every maintained example or tutorial renderer that uses ordinary five-fold regression CV constructs

```python
KFold(n_splits=5, shuffle=True, random_state=0)
```

explicitly.

Within one workflow, path selection, ordinary-PLS comparison, and fixed-model OOF prediction use the
same seeded splitter configuration. The fixed integer seed makes the examples reproducible while
preventing file row order from defining the folds.

Example 03 retains `LeaveOneOut`. Leave-one-out exhaustively holds out every observation once, so a
shuffle setting is neither available nor meaningful.

Grouped, temporal, blocked, spatial, or otherwise structured sampling designs remain
application-owned. Users must replace the demonstration splitter when the data-generating or
sampling process imposes such constraints.

This decision changes example and tutorial results where the former ordered partition affected path
selection. In particular, the Pulp profile at three components selects predictor rank 9 rather than
the former upper-boundary rank 10. Documentation and focused example tests describe the new seeded
shuffled result. Package estimator defaults and numerical algorithms are unchanged.

Historical decision records remain unchanged. This decision supersedes Decisions 0036, 0047, 0062,
0067, and 0085 only where they require ordered or non-shuffled five-fold example partitions. Their
workflow ownership, fold-local preprocessing, comparison, rendering, and validation-scope contracts
otherwise remain in force.

## Consequences

- Maintained K-fold examples are deterministic but not tied to source row order.
- Pi-PLS and ordinary PLS use matched folds in the comparison example.
- Path selection and selection-conditioned OOF prediction use the same explicit fold policy within
  each complete workflow.
- The Pulp tutorial now documents an interior predictor-rank selection under the shuffled folds.
- No package default, estimator API, scoring rule, or mathematical calculation changes.
