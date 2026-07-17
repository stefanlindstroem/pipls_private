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

## Rank selection

`rank_selection.py` asks how adaptive `PiPLSPathCV(search_method="auto")` selects the shared
component count and complete predictor rank when both are declared by the synthetic generator. It
uses three controlled structures: one shared direction, two shared directions, and two shared
directions plus four predictor-only nuisance directions.

Run it from the repository root after installing the package:

```bash
python benchmarks/rank_selection.py
```

It writes `benchmarks/results/rank_selection.csv` with exactly these columns:

```text
scenario
seed
true_n_components
selected_n_components
true_predictor_rank
selected_predictor_rank
test_mse
```

The selected model is refitted on the complete generated training block before independent-test
prediction. Declared ranks are structural references, while selected ranks are predictive
cross-validation choices; exact equality is not an acceptance criterion.

## Predictor-nuisance comparison with ordinary PLS

`predictor_nuisance_comparison.py` asks how paired fixed Pi-PLS and ordinary fixed-component PLS
predict the same independent test block as predictor-specific variation increases. Both methods use
the generator-declared shared rank. Pi-PLS also uses the declared complete predictor-signal rank.

The three scenarios retain two shared directions and vary predictor-only structure:

- `shared_only`: no predictor-specific directions;
- `moderate_predictor_nuisance`: four directions of strengths `(1.5, 1.25, 1.0, 0.75)`;
- `strong_predictor_nuisance`: the same four directions with strengths `(3.0, 2.5, 2.0, 1.5)`.

Run it from the repository root after installing the package:

```bash
python benchmarks/predictor_nuisance_comparison.py
```

It writes `benchmarks/results/predictor_nuisance_comparison.csv` with exactly these columns:

```text
scenario
seed
pipls_test_mse
pls_test_mse
pipls_minus_pls_mse
```

The final column is the paired difference `pipls_test_mse - pls_test_mse`. The benchmark records the
comparison for each generated problem; it does not assert that either method must win.

## Full-versus-randomized solver consistency

`solver_consistency.py` asks whether fixed Pi-PLS predictions and coefficients remain numerically
consistent when only the predictor SVD changes from exact full SVD to randomized truncated SVD. The
models use identical generated data, generator-declared ranks, training-fitted standardization, and
seed.

The three scenarios keep `n_train * n_features = 36864` while changing matrix geometry:

- `wide_n96_p384`;
- `square_n192_p192`;
- `tall_n384_p96`.

Run it from the repository root after installing the package:

```bash
python benchmarks/solver_consistency.py
```

It writes `benchmarks/results/solver_consistency.csv` with exactly these columns:

```text
scenario
seed
prediction_relative_difference
coefficient_relative_difference
```

Both quantities are Frobenius-norm differences relative to the corresponding full-SVD result. The
benchmark records consistency without defining a universal numerical pass threshold and does not
measure runtime.

## Pulp path-selection smoke check

`pulp_path_smoke.py` follows an explicit package-user workflow: it reads the public Pulp
`X.csv` and `Y.csv` tables directly with pandas and fits adaptive `PiPLSPathCV` over component
counts 1 through 4 with five-fold CV and ordered OOF predictions. Because Pulp has only 14
predictors, the script sets `max_predictor_rank=X.shape[1]` so the adaptive search can examine the
complete predictor-rank interval rather than the conservative default rank bound.

Run it from the repository root after installing the data dependencies:

```bash
python benchmarks/pulp_path_smoke.py
```

It writes `benchmarks/results/pulp_path_smoke.csv` with one row and exactly these columns:

```text
selected_n_components
selected_predictor_rank
selection_conditioned_response_standardized_mse
selection_conditioned_pooled_oof_r2
```

The long diagnostic names are intentional. The same CV result selects the ranks and supplies the
reported values, so they are useful workflow diagnostics but not unbiased post-selection or
external-test performance estimates. The script checks that every Pulp row receives exactly one
OOF prediction and fails rather than silently reporting incomplete coverage.

## Sequence status

The four focused synthetic benchmarks and the first separately reviewed real-data smoke check are
implemented. Any additional benchmark must receive its own package-level question, script, and
minimal output contract. High-dimensional real-data checks remain separately reviewed.

Generated outputs under `benchmarks/results/` remain ignored by Git and are excluded from repository
snapshots. Software versions, parallel settings, and timings are included only in a benchmark whose
explicit question requires them. See [`../docs/benchmarks.md`](../docs/benchmarks.md) for
user-facing interpretation and [`../.llm/benchmarking.md`](../.llm/benchmarking.md) for the
normative maintenance contract.
