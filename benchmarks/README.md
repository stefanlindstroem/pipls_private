# Pi-PLS package benchmarks

This directory will contain small, focused validation benchmarks for the long-lived `pipls`
package. It is not a publication-result archive and does not contain manuscript figures, broad
comparison grids, or cached paper outputs.

Each benchmark must answer one programming-user question and write one minimal UTF-8 CSV table.
There is deliberately no universal benchmark runner, manifest, or result schema.

The accepted implementation order is:

1. **Fixed-structure recovery:** fixed Pi-PLS with known ranks; report prediction and latent-subspace
   capture only.
2. **Rank selection:** adaptive Pi-PLS path selection; report true and selected ranks plus test MSE.
3. **Predictor-nuisance comparison:** paired fixed Pi-PLS and ordinary PLS; report their test MSEs
   and paired difference.
4. **Solver consistency:** full versus randomized predictor SVD; report prediction and coefficient
   relative differences.

Generated outputs belong under `benchmarks/results/` and remain ignored by Git. Software versions,
parallel settings, and timings are included only in a benchmark whose explicit question requires
them. See [`../docs/benchmarks.md`](../docs/benchmarks.md) for user-facing interpretation and
[`../.llm/benchmarking.md`](../.llm/benchmarking.md) for the normative maintenance contract.
