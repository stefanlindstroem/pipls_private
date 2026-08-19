# Decision 0157: add a near-saturated synthetic PLS-family stress case

## Status

Accepted; Patch 0157A fixes and tests the synthetic design. Patches 0157B and 0157C remain.

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

Example 03 will gain one deterministic synthetic stress case generated with
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

Add an example-local immutable specification and helper that generate the exact case above. Tests
protect dimensions, latent declarations, noise levels, seed, and deterministic reproduction. This
patch does not add a new plotted case yet.

### 0157B — integrate the stress case into Example 03

Run the cross-covariance Pi-PLS path, least-squares Pi-PLS path, and ordinary-PLS path on one shared
materialized five-fold protocol. Use exhaustive predictor-rank search for both Pi-PLS policies and
write one additional PDF under `examples/results/pls_path_comparison/`. Tests must not assert a
performance ordering between methods.

### 0157C — document the exploratory case and close the decision

Update the maintained example catalogue and user documentation, retain bounded source-distribution
qualification, record the observed fixed-realization behavior factually, and close Decision 0157.

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
- The eventual fourth Example-03 case will increase `make examples` cost modestly but does not need
  to expand clean source-distribution qualification beyond its bounded Pulp branch.
