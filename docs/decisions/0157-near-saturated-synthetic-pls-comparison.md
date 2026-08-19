# Decision 0157: add a near-saturated synthetic PLS-family stress case

## Status

Accepted, implemented, and closed.

## Context

Decision 0156 consolidated the maintained PLS-family component-path comparison into Example 03.
The resulting comparison covers both Pi-PLS response-subspace policies and ordinary PLS on the
three package-owned reference datasets.

A targeted synthetic case is useful for exploring a narrower numerical/statistical question: how
the least-squares/RRR-like response-subspace policy behaves when noisy predictor structure nearly
saturates the information available in each cross-validation training fold. This is exploratory
model-development evidence, not a new package guarantee and not evidence that either response
policy is generally superior.

## Decision

Example 03 includes one deterministic synthetic stress case generated with
`make_pipls_regression()` using the following fixed configuration:

- 25 observations;
- 40 predictors;
- 10 responses;
- 5 shared latent directions;
- 15 predictor-specific latent directions;
- 0 response-specific latent directions;
- Gaussian generator defaults for latent scores;
- common predictor and response noise standard deviation 0.3;
- `random_state=0`.

The configuration is fixed before inspecting comparative results. The repository must not search
multiple seeds, noise levels, dimensions, or realizations to obtain a favorable ordering between
response-subspace policies.

The systematic predictor signal therefore has 20 declared latent directions. Under the maintained
five-fold protocol, each training fold contains 20 observations and centered predictor data can
have numerical rank at most 19. The intended stress mechanism is thus the near-saturation of the
foldwise predictor-rank domain, not an assertion that the five shared directions themselves are
close to the sample count.

## Patch sequence

### 0157A — fixed design and deterministic regression coverage

Implemented. An example-local immutable specification and helper generate the exact case above.
Tests protect dimensions, latent declarations, noise levels, seed, and deterministic reproduction.
The design was fixed before comparative results were inspected.

### 0157B — integrate the stress case into Example 03

Implemented. Example 03 now runs the cross-covariance Pi-PLS path, least-squares Pi-PLS path, and
ordinary-PLS path on one shared materialized five-fold protocol for the fixed synthetic case. Both
Pi-PLS policies use exhaustive predictor-rank search. The additional figure is written to
`examples/results/pls_path_comparison/synthetic_stress_component_path_comparison.pdf`. Focused
regression coverage checks matched folds, exhaustive coverage, the complete 1--10 component domain,
finite CV-MSE paths, and the expected artifact path without asserting a performance ordering.

### 0157C — document the exploratory case and close the decision

Implemented. The maintained example catalogue and user documentation now include the fourth case,
while source-distribution qualification remains bounded to the Pulp branch of Example 03. The
observed fixed-realization behavior is recorded below and Decision 0157 is closed.

## Observed fixed-realization behavior

The fixed `random_state=0` realization does not show a broad degradation of the least-squares
response-subspace policy. The mean response-standardized CV-MSE minima are:

- cross-covariance Pi-PLS: 0.9036 at 5 components;
- least-squares Pi-PLS: 0.9022 at 5 components;
- ordinary PLS: 0.9373 at 8 components.

Across the 1--10 component path, the two Pi-PLS curves remain close and alternate in which policy
has the lower mean CV-MSE. At 10 components their mean CV-MSE values coincide in this run, consistent with the fitted-map
limiting case when the component count reaches the 10-dimensional response space. These values are exploratory
model-development evidence from one predeclared realization; they neither establish equivalence nor
support a general superiority claim for either response policy.

## Relationship to earlier decisions

- Decision 0015 continues to own the public synthetic-data API. This decision uses the existing
  configurable generator and introduces no new package-level generator surface.
- Decision 0045 continues to own the comparison-only boundary for ordinary PLS.
- Decision 0155 continues to own the two Pi-PLS response-subspace policies and their publication
  boundary.
- Decision 0156 continues to own Example 03 as the sole maintained PLS-family comparison workflow.

## Consequences

- The exploratory geometry is reproducible and reviewable before its comparative result is known.
- No public runtime API changes are required.
- The fourth Example-03 case increases `make examples` cost modestly but does not expand clean
  source-distribution qualification beyond its bounded Pulp branch.
- The predeclared stress hypothesis was useful even though this realization did not show the
  expected broad least-squares deterioration; retaining the result avoids post-hoc seed selection.
