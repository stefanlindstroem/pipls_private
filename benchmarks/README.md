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

## Pulp component-path smoke check

`pulp_path_smoke.py` follows an explicit package-user workflow: it reads the public Pulp
`X.csv` and `Y.csv` tables directly with pandas and evaluates component counts 1 through 4 with
`PiPLSPathCV(refit=False)`. Predictor rank is selected conditionally for each component count using
the ordinary `samples_per_predictor_rank=5`, `cv=5`, and adaptive-search defaults.

Run it from the repository root after installing the data dependencies:

```bash
python benchmarks/pulp_path_smoke.py
```

It writes `benchmarks/results/pulp_path_smoke.csv` with four rows and exactly these columns:

```text
n_components
predictor_rank
predictor_rank_policy
response_standardized_cv_mse_mean
response_standardized_cv_mse_fold_sd
n_splits
```

The numeric predictor rank is present in every row. The ordinary workflow records
`predictor_rank_policy=optimized`. The fold SD describes variation across the five validation
folds; it is not a confidence interval or an independent standard error. The benchmark does not
choose a final component count, refit a final model, or generate a figure.

## Sugarcane high-dimensional component-path smoke check

`sugarcane_path_smoke.py` applies the same component-path workflow to the transparent 57-row,
1,721-predictor Sugarcane tables. It therefore exercises the ordinary public defaults when
$p \gg n$ without setting a predictor-rank ceiling or forcing an SVD solver.

Run it from the repository root after installing the data dependencies:

```bash
python benchmarks/sugarcane_path_smoke.py
```

It writes `benchmarks/results/sugarcane_path_smoke.csv` with four rows and the same six-column
component-path contract used by Pulp. The benchmark verifies deterministic finite mean CV-MSE and
fold SD values, one numeric predictor rank per component count, and ordered rows. It does not select
spectral preprocessing, choose the final model, report runtime, or generate a PDF.

## Tobacco randomized-SVD component-path smoke check

`tobacco_path_smoke.py` reads the transparent 347-row, 1,557-predictor Tobacco tables and evaluates
component counts 1 through 8. It uses an explicit `PiPLSRegression(svd_solver="randomized",
random_state=0)` template and `PiPLSPathCV(search_method="auto", refit=False)`, so the example
demonstrates randomized predictor decomposition together with adaptive conditional predictor-rank
scanning.

Run it from the repository root after installing the example dependencies:

```bash
python benchmarks/tobacco_path_smoke.py
```

It writes `benchmarks/results/tobacco_path_smoke.csv` with eight rows and the same six-column
component-path contract as Pulp and Sugarcane. The upper component count is a bounded example
choice, not a claim that the complete 13-component path has been exhausted. The benchmark does not
choose a final model, generate a PDF, compare SVD solvers, report timing, or select spectral
preprocessing.

## Sequence status

The four focused synthetic benchmarks and the Pulp, Sugarcane, and Tobacco real-data smoke checks
are implemented. Any additional benchmark must receive its own package-level question, script, and
minimal output contract.

Generated outputs under `benchmarks/results/` remain ignored by Git and are excluded from repository
snapshots. Software versions, parallel settings, and timings are included only in a benchmark whose
explicit question requires them. See [`../docs/benchmarks.md`](../docs/benchmarks.md) for
user-facing interpretation and [`../.llm/benchmarking.md`](../.llm/benchmarking.md) for the
normative maintenance contract.
