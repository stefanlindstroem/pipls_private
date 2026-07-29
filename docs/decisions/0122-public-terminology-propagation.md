# Decision 0122: public terminology propagation

## Status

Accepted and implemented.

## Context

Decision 0121 established precise names for the fixed Pi-PLS construction, but much of the living
public surface still used older wording. README material described paired predictor and response
latent variables, path pages treated `n_components` as an undefined generic component count, and
some generated docstrings called $P$ and $Q$ rotations as though that were their primary
mathematical name. The inspection guide also retained lower-case $q$ in element notation and could
be read as conflating Pi-PLS directions with reconstruction loadings.

The established Python identifiers containing `rotations` must remain stable. The familiar words
“component path” and `n_components` must also remain available because they are public API and
cross-method terminology. Propagation therefore needs to improve explanatory prose without
renaming objects, changing equations, or rewriting historical decision records.

## Decision

Propagate Decision 0121 through the living public documentation and generated source docstrings:

- introduce Pi-PLS through **paired latent modes** rather than paired latent variables;
- define public `n_components` as the number of paired latent modes $h$ and public
  `predictor_rank` as the retained predictor-subspace dimension $r_\pi$ wherever users first meet
  the two-rank interface;
- call $P$ and $Q$ orthonormal predictor and response **directions** in mathematical prose;
- describe $d_k=D_{kk}$ as the dilation of paired latent mode $k$;
- write weighted response-direction columns as $d_kQ_{:k}$ and retain the array name $QD$;
- state explicitly that public fields named `predictor_rotations` and `response_rotations` retain
  their identifiers for compatibility while containing the canonical directions;
- keep X and Y reconstruction loadings distinct from Pi-PLS directions;
- retain “component path,” “component count,” and related generic PLS-family wording where they
  name an established API object, a plot type, or a cross-method comparison, but define their
  Pi-PLS meaning at the owning entry point.

Apply the wording to the README, documentation home, generated API introductions, path and
inspection guides, tutorials, example catalogue and comments, and public factorization/result
source docstrings. Do not rewrite historical decisions merely to modernize their vocabulary.
Synthetic loading matrices and generic PLS-family score/loadings terminology remain loadings and
components where those are the correct objects.

## Consequences

- New users encounter one vocabulary across onboarding, API reference, tutorials, examples, and
  generated docstrings.
- Public Python identifiers and result shapes remain unchanged.
- The distinction between Pi-PLS directions and reconstruction loadings is explicit.
- Existing component-path and ordinary-PLS comparison language remains readable and familiar.
- Estimator, search, validation, synthetic-generator, benchmark, rendering, example, and real-data
  behavior are unchanged.
