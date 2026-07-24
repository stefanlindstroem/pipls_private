# Decision 0104: new-user documentation route

## Status

Accepted.

## Context

The README and documentation home page explained Pi-PLS mechanics and linked the two tutorial
workflows, but they did not state when separate predictor-subspace and predictive-component ranks
may be useful. Both tutorial openings also exposed maintained source paths, renderer ownership, and
figure-generation commands before the reader reached the modeling workflow.

Those details are required for reproducibility, but they are maintainer-facing provenance rather
than the first information a new user needs.

## Decision

The README and served home page give a restrained application-oriented motivation: Pi-PLS targets
multivariate responses and predictor blocks whose structured variation need not all be predictive,
and its two ranks let users examine the predictor subspace and paired predictive relation
separately. The text makes no general performance claim and retains ordinary PLS and other
multivariate methods as problem-dependent alternatives.

Each tutorial now opens in this order:

1. problem and purpose;
2. coverage;
3. setup or data;
4. modeling workflow.

Maintained source paths, snippet provenance, renderer ownership, and documentation-generation
commands move to a terminal `Reproduce this tutorial` section. Installation instructions remain in
the setup section when they are needed to run the workflow.

## Consequences

New readers encounter the modeling question before repository maintenance machinery. Reproducible
source ownership and strict documentation-build instructions remain visible without interrupting
the tutorial route. The change affects documentation organization only; it changes no estimator,
numerical result, example calculation, or generated figure.
