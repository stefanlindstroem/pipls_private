# Pi-PLS package benchmarks

This directory contains lightweight validation contracts for the long-lived `pipls` package. It is
not a publication-result archive and does not contain manuscript figures, complete comparison
grids, or cached paper outputs.

The first accepted suite is [`manifests/synthetic-v1.yaml`](manifests/synthetic-v1.yaml). It defines
controlled synthetic scenarios, deterministic seeds, method roles, runtime tiers, metrics,
tolerance policy, and generated-result format. The result record schema is
[`schema/result-v1.schema.json`](schema/result-v1.schema.json).

The CI tier is implemented by [`run_synthetic.py`](run_synthetic.py). Run it through:

```bash
make benchmark-ci
```

This writes `benchmarks/results/synthetic-ci.jsonl`. Generated outputs are ignored by Git unless a
later decision explicitly freezes a small package-validation fixture with documented meaning and
tolerances. The standard and performance tiers remain unimplemented.

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


## Execution boundary

The runner is repository-local and is not exported from `pipls`. It consumes the manifest rather
than duplicating scenario values. Adaptive path candidates use joblib threads with one native
linear-algebra thread per candidate so the CI tier stays small and avoids nested oversubscription.
Resource timings are descriptive and are excluded from repeatability comparisons.
