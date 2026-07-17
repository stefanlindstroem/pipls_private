# Pi-PLS package benchmarks

This directory contains lightweight validation contracts for the long-lived `pipls` package. It is
not a publication-result archive and does not contain manuscript figures, complete comparison
grids, or cached paper outputs.

The first accepted suite is [`manifests/synthetic-v1.yaml`](manifests/synthetic-v1.yaml). It defines
controlled synthetic scenarios, deterministic seeds, method roles, runtime tiers, metrics,
tolerance policy, and generated-result format. The flat CSV record schema is
[`schema/result-v2.schema.json`](schema/result-v2.schema.json).

The CI tier is implemented by [`run_synthetic.py`](run_synthetic.py). Run it through:

```bash
make benchmark-ci
```

This writes `benchmarks/results/synthetic-ci.csv`. The file is ordinary UTF-8, comma-delimited
tabular data with one header row and one result row per scenario, seed, and method. Open it directly
with pandas, R, spreadsheet software, or a text editor. Empty cells mean that a field does not apply
to that method or was not measured.

For example:

```python
import pandas as pd

results = pd.read_csv("benchmarks/results/synthetic-ci.csv")
print(results.to_string(index=False))
```

Generated outputs are ignored by Git unless a later decision explicitly freezes a small
package-validation fixture with documented meaning and tolerances. The standard and performance
tiers remain unimplemented.

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


## Human and machine readability

Benchmark results are flat CSV because they are tables. Column order, names, types, null handling,
and schema version are defined by the machine-readable JSON schema. This gives humans a familiar
file and machines an explicit contract without requiring nested-record parsing.
