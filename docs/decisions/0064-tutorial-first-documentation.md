# Decision 0064: tutorial-first documentation

## Status

Accepted.

## Context

The public site already contained concise guides and generated API reference, but a new user still
had to combine several pages to understand one complete Pi-PLS analysis. Decisions 0062 and 0063
established a canonical executable Pulp workflow and deterministic figures generated from it.

## Decision

Use the Pulp analysis as the primary pedagogical route through the served documentation.

`docs/tutorials/pulp.md` follows one model-development sequence from data loading through pipeline
construction, component-path evaluation, explicit fixed-rank fitting, out-of-fold prediction, and
fitted-model interpretation. It includes each generated chart separately and links every chart to
the corresponding inspection explanation, plotting reference, and theory section where relevant.

Executable code excerpts are included from `examples/_support/pulp_workflow.py` and
`examples/10_pulp_real_data.py` through checked `pymdownx.snippets` sections. The tutorial does not
maintain a second numerical implementation.

The site home page directs new users to the tutorial. The quickstart remains a compact fixed-fit API
introduction, while the detailed guide and reference pages retain their general roles.

The tutorial states the validation provenance of its prediction figures. Its fixed-parameter OOF
predictions are selection-conditioned because the same observations were used earlier to inspect the
component path; they are not presented as independent post-selection performance estimates.

## Consequences

The complete Pulp workflow can be read in one linear document without hiding the reusable reference
material. Code, figures, and analysis results remain tied to repository execution. Nested tutorial
links may use `../` only when they resolve inside `docs/`; repository tests enforce that boundary.

The following patch may consolidate surrounding guides now that the tutorial owns the worked
example. This decision does not change package behavior or the numerical Pulp workflow.
