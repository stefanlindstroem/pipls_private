# Decision 0065: consolidate tutorial, guide, and reference ownership

## Status

Accepted.

## Context

Decision 0064 made the Pulp tutorial the primary pedagogical route, but several older pages still
repeated the same component-selection narrative, Pulp report details, and plotting examples. The
largest overlap was in `model_inspection.md`, which mixed general scientific definitions with
example-specific report composition and artifact descriptions.

## Decision

Separate the served documentation into three roles:

1. the Pulp tutorial owns the complete linear worked analysis and its generated figures;
2. guide pages own concise task-oriented procedures such as fixed fitting, parameter selection, and
   cross-validation;
3. reference pages own general scientific definitions, exact search behavior, and generated API
   signatures.

`model_inspection.md` becomes the general interpretation reference. It defines the immutable result
objects and gives one stable section for every maintained plot without repeating the Pulp,
Sugarcane, or Tobacco report implementation. The tutorial links each figure to the matching stable
section and exact plotting API object.

`parameter_selection.md` owns the short two-stage procedure. `path_analysis.md` owns detailed search
surface, rank policy, pipeline, refit, and diagnostic behavior. `examples.md` is a catalogue of
maintained scripts and outputs rather than a second tutorial. The plotting API remains operational
and delegates scientific interpretation to `model_inspection.md`.

Repository-generated Pulp images appear only in the tutorial. Other pages link to the tutorial or
the general interpretation reference rather than embedding or retelling the worked analysis.

## Consequences

A new user can follow one complete analysis without encountering competing explanations. A user
seeking a specific rule or function can move directly to a concise guide or reference page. Stable
inspection anchors protect tutorial links while narrative wording remains free to evolve.

The tutorial-first documentation phase is complete. The next increment is first-release definition
and metadata. This decision changes documentation ownership only; numerical, estimator, plotting,
example, and generated-asset behavior remain unchanged.
