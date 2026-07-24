# Decision 0105: documentation and implementation alignment

## Status

Accepted.

## Context

The first-release hardening and new-user onboarding series changed several public contracts in
quick succession: fixed ranks became required, path selection became selection-only by default, the
default scorer became a stable string, public result records gained stronger invariants, numerical
rank became fold-aware, examples were renumbered, and optional dependency groups were simplified.

The generated API docstrings reflected those changes, but some maintained examples and explanatory
pages still repeated `refit=False` instead of demonstrating the new default. The fixed-regression
reference also summarized the decomposition without listing its numerical-rank and resolved-solver
fields, and the guide layer incorrectly said that the Pulp workflow obtained its conditional rank
profile directly from `cv_results_`.

## Decision

Current user-facing selection workflows construct `PiPLSPathCV()` without spelling the default
`refit=False`. Text continues to state that the default is selection-only, and `refit=True` remains
explicit wherever automatic global-best refitting is intended. Historical decision records retain
the syntax that was current when those decisions were accepted.

The fixed-regression reference lists every public decomposition quantity, including numerical-rank
exactness, rank tolerance, resolved predictor solver, and the derived standardized regression map.
It also includes the fixed estimator's public `set_output()` method. Path-selection details identify
`path_search_exhaustive_` as the public indicator of whether adaptive search evaluated every
admissible pair.

Guide-layer descriptions of the Pulp workflow use the public
`predictor_rank_profile()` method rather than direct `cv_results_` column access. Reproducibility
text describes distributed code rather than implying that the unreleased package has already been
released.

## Consequences

The maintained examples, tutorial renderers, prose, generated API configuration, and guide layer
now demonstrate the same defaults and public surfaces as the implementation. The patch changes no
selection result, estimator fit, figure, public signature, or numerical behavior.
