# Pi-PLS package benchmarks

This directory contains small, focused validation benchmarks for the long-lived `pipls` package.
It is not a publication-result archive and does not contain manuscript figures, broad comparison
grids, or cached paper outputs.

Each benchmark answers one programming-user question and writes one minimal UTF-8 CSV table. There
is deliberately no universal benchmark runner, manifest, or result schema.

## Fixed-structure recovery

`fixed_structure_recovery.py` asks whether fixed-rank Pi-PLS recovers known shared and complete
signal subspaces while predicting an independent synthetic test block. It uses the public
`make_pipls_train_test` generator and fixed oracle ranks.

The two scenarios are:

- `shared_only`: two shared latent directions and no predictor-specific directions;
- `predictor_specific_nuisance`: the same observed dimensions, sample sizes, shared strengths,
  response structure, and noise level, plus four predictor-only directions.

Run it from the repository root after installing the package:

```bash
python benchmarks/fixed_structure_recovery.py
```

It writes `benchmarks/results/fixed_structure_recovery.csv` with exactly these columns:

```text
scenario
seed
test_mse
predictor_shared_capture
predictor_signal_capture
response_shared_capture
```

The subspace captures are mean squared canonical correlations. Predictor and response truth bases
are first expressed in the estimator coordinates using the observed-variable scales from the
generator and the training scales learned by the fitted model.

## Remaining sequence

The next separately reviewed benchmarks are rank selection, predictor-nuisance comparison with
ordinary PLS, and full-versus-randomized solver consistency.

Generated outputs under `benchmarks/results/` remain ignored by Git and are excluded from repository
snapshots. Software versions, parallel settings, and timings are included only in a benchmark whose
explicit question requires them. See [`../docs/benchmarks.md`](../docs/benchmarks.md) for
user-facing interpretation and [`../.llm/benchmarking.md`](../.llm/benchmarking.md) for the
normative maintenance contract.
