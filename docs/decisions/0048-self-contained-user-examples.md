# Decision 0048: require self-contained user examples

## Status

Accepted and implemented.

## Context

Numbered examples are user-facing entry points. They must demonstrate a recognizable use case,
comparison, or benchmark without requiring knowledge of a publication or earlier project history.
The former advanced-cross-validation example combined grouped, leave-one-out, and temporal
splitters on an unexplained random matrix. It displayed API capabilities, but not a coherent task a
programming user could adapt. The synthetic-data example also generated a structured problem but
printed only two unlabeled numbers, leaving the data construction and result meaning unclear.

Brevity remains important, but removing labels and explanation is not useful simplification.

## Decision

- Remove `examples/07_advanced_cv.py`. Grouped, leave-one-out, and temporal validation remain
  supported and documented in `docs/cross_validation.md`; a future numbered example requires a
  concrete sampling use case.
- Keep `examples/08_synthetic_data.py` as a self-contained train/test example. Explain the roles of
  shared, predictor-specific, and response-specific latent directions in comments and labeled
  output.
- Print labeled matrix dimensions, known latent structure, fitted-model choices, prediction shape,
  and independent-test $R^2$ rather than bare values.
- Require numbered examples to be understandable without references to a paper, manuscript, or
  downstream reproduction repository.
- Treat numbered examples as minimal use cases, explicit comparisons, or focused benchmarks. An API
  capability without an interpretable problem belongs in user documentation or focused tests.
- Preserve the concise-example rules from Decision 0046: avoid one-use scaffolding, redundant
  validation of repository-owned data, and production-style filesystem handling.

## Consequences

The numbered sequence now moves from one literal fixed fit to an explained synthetic train/test
problem, an explicit PLS comparison, and complete real-data Pi-PLS analyses. Advanced splitter
support remains available without occupying a context-free numbered example. Focused tests verify
the synthetic example's labeled output and prevent publication-dependent language from returning to
numbered scripts.
