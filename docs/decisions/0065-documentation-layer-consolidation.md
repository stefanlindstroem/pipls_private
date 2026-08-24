# Decision 0065: consolidate tutorial, guide, and reference ownership

## Status

Accepted and implemented.

## Context

An earlier tutorial-first increment made the Pulp tutorial the primary pedagogical route, but
several older pages still repeated the same component-selection narrative, Pulp report details, and
plotting examples. The
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
section and exact numerical result object or field.

`path_analysis.md` owns detailed search surface, rank policy, pipeline, refit, and OOF behavior.
`examples.md` is a catalogue of maintained scripts and outputs rather than a second tutorial.
`model_inspection.md` owns scientific interpretation of immutable numerical results; rendering is
caller-owned and no runtime plotting API exists.

Repository-generated tutorial images appear only in their owning tutorials. Other pages link to
the tutorial or the general interpretation reference rather than embedding or retelling the worked
analysis.

Cross-references are semantic rather than mechanical. A substantive dataset reference should lead
to the maintained dataset detail, a publication reference to the canonical citation section, and a
mathematical concept to the most specific maintained theory section when that route helps the
reader. Immediate repetitions, headings, code, figure alt text, and prose already on the canonical
destination need not be linked. Strict documentation builds own link-resolution validation; pytest
must not freeze particular prose-link placement or require every served page to satisfy an
incidental hyperlink-count rule.

The served site is a strict self-contained MkDocs build. Generated API pages cover the documented
public modules and derive signatures and field documentation from audited docstrings. Numbered
decisions and `.llm` are maintainer records and are excluded from served navigation and search.
Documentation inputs required for a clean build ship in the source distribution, and CI validates
the strict site both from the checkout and from an isolated source-distribution installation. The
canonical rendered site is deployed from the repository rather than committed as generated `site/`
output.

## Consequences

A new user can follow one complete analysis without encountering competing explanations. A user
seeking a specific rule or function can move directly to a concise guide or reference page. Stable
inspection anchors protect tutorial links while narrative wording remains free to evolve.

The documentation layer has one owner for each type of content and one strict build boundary.
Generated reference, tutorial assets, source-distribution inputs, and deployment are validation
concerns rather than competing narrative layers. This decision changes documentation ownership
only; numerical and estimator behavior remain unchanged.
