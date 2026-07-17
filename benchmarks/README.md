# Pi-PLS package benchmarks

This directory contains lightweight validation contracts for the long-lived `pipls` package. It is
not a publication-result archive and does not contain manuscript figures, complete comparison
grids, or cached paper outputs.

The first accepted suite is [`manifests/synthetic-v1.yaml`](manifests/synthetic-v1.yaml). It defines
controlled synthetic scenarios, deterministic seeds, method roles, runtime tiers, metrics,
tolerance policy, and generated-result format. The result record schema is
[`schema/result-v1.schema.json`](schema/result-v1.schema.json).

No benchmark runner or numerical result fixture is introduced by the contract patch. The next
increment may implement the CI tier against this versioned manifest. Generated outputs belong under
`benchmarks/results/` and are ignored by Git unless a later decision explicitly freezes a small
package-validation fixture with documented meaning and tolerances.

## Scope

The suite validates programming-user concerns:

- prediction on independent generated test observations;
- Pi-PLS rank-selection behavior;
- recovery of known latent subspaces;
- deterministic and solver-consistency behavior;
- representative runtime and optional memory measurements.

Ordinary `sklearn.cross_decomposition.PLSRegression` is the external comparator because it is the
nearest user-facing baseline. OLS, CCA, broad simulation grids, and figure generation remain outside
this repository unless a later package-level identity requires a narrowly scoped check.

## Standardization boundary

Neither generated block is standardized before model fitting. Pi-PLS and ordinary PLS learn their
own centering and scaling from the training observations. Any inner cross-validation learns those
statistics separately within each training fold and refits them on the complete generated training
block after selection.
