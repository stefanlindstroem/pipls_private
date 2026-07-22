# Decision 0074: tutorial-owned user workflow

## Status

Accepted and implemented.

## Context

The Pulp tutorial now contains the complete normal Pi-PLS workflow: data loading, component-path
inspection, conditional predictor-rank selection, fixed fitting, OOF diagnostics, and model
interpretation. It also has enough context to explain common variations such as direct fixed fitting,
custom splitters and scorers, pipelines, and automatic refitting.

Several older pages still repeated shorter versions of the same material. `quickstart.md`,
`estimator_api.md`, `parameter_selection.md`, and `preprocessing.md` divided one user journey across
multiple navigation groups and duplicated information already present in the tutorial, generated API
reference, or estimator docstrings.

## Decision

The Pulp tutorial is the sole linear description of normal Pi-PLS use. It includes a compact common-
variations section covering direct fixed fitting, nondefault validation and scoring, fold-local
pipelines, and `refit=True`.

The home page contains only a minimal fixed-fit example and routes readers to the tutorial or exact
reference pages. The generated fixed-regression and path-selection pages own estimator parameters,
preprocessing, solver behavior, fit-state safety, fitted results, method contracts, and path-result
objects.

`path_analysis.md` is retained only as an advanced search-behavior reference. It owns nondefault
component and predictor-rank requests, search policies, rank ceilings, tie-breaking, pipelines,
refitting, and detailed candidate surfaces. `cross_validation.md` remains the specialized splitter,
scoring, OOF, and leave-one-out reference. `model_inspection.md` remains the stable scientific
interpretation reference.

Delete the redundant standalone pages:

```text
quickstart.md
estimator_api.md
parameter_selection.md
preprocessing.md
```

Flatten the served navigation to Home, Tutorial, Examples, Reference, Data and validation, and
Scientific background. Remove editor-backup files from the documentation tree.

## Consequences

A new user follows one complete analysis rather than assembling the normal workflow from several
short guides. Exact estimator behavior remains discoverable beside the generated signatures, while
advanced path and validation behavior retain dedicated references. The served documentation loses
four pages and substantially reduces duplicated prose without changing package behavior, examples,
figures, or public APIs.

This decision supersedes Decision 0065 where that record assigns routine procedures to separate task
guides. Decision 0065 remains the historical basis for keeping the tutorial, scientific
interpretation reference, and generated API contracts distinct.
