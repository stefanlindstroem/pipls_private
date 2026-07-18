# Benchmarking contract

## Purpose

Package benchmarks must answer narrow questions that matter to programming users. A benchmark is
not a general experiment runner, publication simulation grid, or infrastructure exercise. Each
benchmark must have one stated question, one controlled setup, and one small result table whose
columns are needed to answer that question.

This file is normative for benchmark design. No benchmark implementation should be added until its
question, comparison, metrics, output columns, and interpretation are accepted separately.

## Design rule: one benchmark, one question

Every benchmark must define:

1. the user-facing question;
2. the synthetic structure varied and held fixed;
3. the fitted method or paired methods;
4. the minimum metrics needed to answer the question;
5. one dedicated CSV output with no unrelated columns;
6. how the result should and should not be interpreted.

Do not create a universal row schema, a universal manifest, or a single runner that combines
prediction, rank selection, subspace recovery, solver consistency, software versions, parallelism,
and timings. Those are different questions and must remain separate.

CSV remains the default for tabular output. Column names must be explicit, values must be directly
readable with pandas, R, spreadsheet software, or a text editor, and empty columns must not be added
for metrics that do not apply.

## Common scientific boundary

Use `pipls.datasets.make_pipls_train_test` for controlled synthetic data. Training and test blocks
share latent loadings, strengths, and observed-variable scales but have independent scores and
noise. A seed identifies the generated problem.

Do not fit centering, scaling, or another learned transform before model fitting. Pi-PLS and
ordinary PLS learn their statistics from the training data supplied to each fit. Any inner
cross-validation must fit the complete candidate separately within each training fold and refit the
selected candidate on the complete generated training block.

Ordinary `PLSRegression` is the only planned external comparator because it is the nearest
programming-user baseline. OLS, CCA, publication grids, figure generation, and manuscript claims
remain outside this repository.

Real-data component-path analyses belong under `examples/`. They read repository `X.csv` and
`Y.csv` tables directly, use public estimator calls, and label diagnostics according to their actual
validation protocol. They are user-run analyses rather than benchmark or default-test jobs.

## Focused benchmarks

### 1. Fixed-structure recovery

**Question:** When the true shared dimension and predictor-signal rank are supplied, does fixed
Pi-PLS recover the intended latent subspaces and predict independent responses?

**Method:** fixed `PiPLSRegression` only, with `scale=True`, `svd_solver="full"`,
`n_components` equal to the generator-declared shared rank, and `predictor_rank` equal to the
complete generator-declared predictor-signal rank. No cross-validation is performed.

**Implemented scenarios:**

- `shared_only`: two shared directions and no predictor-specific directions;
- `predictor_specific_nuisance`: the same sample sizes, observed dimensions, shared strengths,
  response structure, and noise level, plus four predictor-only directions.

Both scenarios use 160 training samples, 160 test samples, 24 predictors, six responses, no
response-specific directions, shared strengths `(2.5, 1.5)`, noise `(0.2, 0.2)`, and seeds 1729,
2718, and 3141. The nuisance strengths are `(3.0, 2.5, 2.0, 1.5)`.

**Metrics:**

- `test_mse` is the mean squared residual over all test samples and responses in original response
  units;
- `predictor_shared_capture` compares the true predictor-shared subspace with fitted $P$;
- `predictor_signal_capture` compares the complete true predictor-signal subspace with fitted
  $\Pi$;
- `response_shared_capture` compares the true response-shared subspace with fitted $Q$.

For true basis $A$ and estimated basis $B$, with orthonormal column bases $Q_A$ and $Q_B$, capture
is $\lVert Q_A^{\mathsf{T}} Q_B \rVert_{\mathrm{F}}^2 / \dim[\mathrm{col}(A)]$. Generator
loadings are mapped into the estimator coordinates using `truth.feature_scale / model.x_scale_` for
predictors and `truth.target_scale / model.y_scale_` for responses. Learned scales therefore come
only from the benchmark training block.

**Output:** `benchmarks/results/fixed_structure_recovery.csv`.

Required columns:

- `scenario`;
- `seed`;
- `test_mse`;
- `predictor_shared_capture`;
- `predictor_signal_capture`;
- `response_shared_capture`.

The benchmark is implemented by `benchmarks/fixed_structure_recovery.py`. No selected-rank fields,
software versions, timings, ordinary PLS comparison, solver comparison, or figures belong in this
table.

### 2. Rank selection

**Question:** When the generator declares the shared dimension and complete predictor-signal rank,
which ranks does `PiPLSPathCV(search_method="auto")` select for prediction?

**Method:** adaptive Pi-PLS path selection only. Candidate models use
`PiPLSRegression(scale=True, svd_solver="full")`, five-fold CV, a benchmark-specific
rule-based predictor-rank bound with `samples_per_predictor_rank=10.0`, and one execution job.
Every candidate
learns model centering and scaling inside its training fold, and the selected model refits on the
complete generated training block.

**Implemented scenarios:**

- `one_shared`: one shared direction of strength `(2.5,)` and no predictor-specific directions;
- `two_shared`: two shared directions of strengths `(2.5, 1.5)` and no predictor-specific
  directions;
- `two_shared_with_predictor_nuisance`: the same two shared directions plus four predictor-only
  directions of strengths `(3.0, 2.5, 2.0, 1.5)`.

All scenarios use 160 training samples, 160 test samples, 24 predictors, six responses, no
response-specific directions, noise `(0.2, 0.2)`, and seeds 1729, 2718, and 3141.

**Metrics:**

- `true_n_components` is `truth.n_shared`;
- `selected_n_components` is `PiPLSPathCV.best_n_components_`;
- `true_predictor_rank` is `truth.n_shared + truth.n_predictor_specific`;
- `selected_predictor_rank` is `PiPLSPathCV.best_predictor_rank_`;
- `test_mse` is the mean squared residual over the independent test samples and responses in
  original response units after full-training refit.

**Output:** `benchmarks/results/rank_selection.csv`.

Required columns:

- `scenario`;
- `seed`;
- `true_n_components`;
- `selected_n_components`;
- `true_predictor_rank`;
- `selected_predictor_rank`;
- `test_mse`.

The declared ranks are structural references, while selected ranks optimize a finite
cross-validation estimate of predictive loss. Exact equality, a maximum rank error, and a
predictive pass threshold are not benchmark acceptance criteria. Fixed-oracle comparison, ordinary
PLS, exhaustive-versus-adaptive comparison, subspace metrics, solver comparison, candidate counts,
software versions, timings, and figures do not belong in this table.

The benchmark is implemented by `benchmarks/rank_selection.py`.

### 3. Predictor-nuisance comparison with PLS

**Question:** When predictor-specific variation increases, how does fixed Pi-PLS prediction compare
with ordinary fixed-component PLS on the same generated train/test problem?

**Methods:** paired fixed Pi-PLS and `PLSRegression`, both with `scale=True` and the
generator-declared shared dimension as `n_components`. Pi-PLS additionally uses exact predictor SVD
and the declared complete predictor-signal rank. Both methods fit their own centering and scaling
statistics from the complete generated training block; no cross-validation is performed.

**Implemented scenarios:**

- `shared_only`: two shared directions and no predictor-specific directions;
- `moderate_predictor_nuisance`: the same shared structure plus four predictor-only directions of
  strengths `(1.5, 1.25, 1.0, 0.75)`;
- `strong_predictor_nuisance`: the same four predictor-only directions with strengths
  `(3.0, 2.5, 2.0, 1.5)`.

All scenarios use 160 training samples, 160 test samples, 24 predictors, six responses, no
response-specific directions, shared strengths `(2.5, 1.5)`, noise `(0.2, 0.2)`, and seeds 1729,
2718, and 3141. The two nuisance scenarios keep the nuisance dimension fixed so their difference is
its strength rather than another rank change.

**Metrics:**

- `pipls_test_mse` is the mean squared Pi-PLS residual over all independent test samples and
  responses in original response units;
- `pls_test_mse` is the corresponding ordinary PLS quantity;
- `pipls_minus_pls_mse` is `pipls_test_mse - pls_test_mse`, so negative values favor Pi-PLS for that
  generated problem and positive values favor ordinary PLS.

**Output:** `benchmarks/results/predictor_nuisance_comparison.csv`.

Required columns:

- `scenario`;
- `seed`;
- `pipls_test_mse`;
- `pls_test_mse`;
- `pipls_minus_pls_mse`.

Use a wide paired table because the comparison itself is the question. The benchmark is implemented
by `benchmarks/predictor_nuisance_comparison.py`. It does not define a superiority threshold or add
rank selection, latent recovery, solver comparison, timings, software metadata, figures, OLS, or
CCA.

### 4. Solver consistency

**Question:** For a fixed high-dimensional Pi-PLS model, are full and randomized predictor SVD
numerically consistent for the same generated data and seed?

**Methods:** paired fixed `PiPLSRegression` fits with `scale=True`, generator-declared shared and
complete predictor-signal ranks, and respectively `svd_solver="full"` and
`svd_solver="randomized"`. The generated training block, independent test block, fitted model
ranks, and seed are identical within each pair. Only the first predictor decomposition differs;
response and coupling decompositions remain exact in both fits.

**Implemented scenarios:**

- `wide_n96_p384`: 96 training samples and 384 predictors;
- `square_n192_p192`: 192 training samples and 192 predictors;
- `tall_n384_p96`: 384 training samples and 96 predictors.

Each training predictor matrix contains 36,864 entries. All scenarios use 96 independent test
samples, eight responses, three shared directions of strengths `(3.0, 2.0, 1.5)`, five
predictor-specific directions of strengths `(2.5, 2.0, 1.5, 1.0, 0.75)`, no response-specific
directions, noise `(0.3, 0.2)`, and seeds 1729, 2718, and 3141.

**Metrics:**

- `prediction_relative_difference` is the Frobenius norm of the difference between randomized- and
  full-SVD predictions on the independent test block, divided by the Frobenius norm of the full-SVD
  predictions;
- `coefficient_relative_difference` is the corresponding relative Frobenius difference between the
  fitted `coef_` arrays in original response units.

The full-SVD result is the reference path. A machine-epsilon denominator floor handles a
zero-reference edge case without adding another output field.

**Output:** `benchmarks/results/solver_consistency.csv`.

Required columns:

- `scenario`;
- `seed`;
- `prediction_relative_difference`;
- `coefficient_relative_difference`.

The benchmark is implemented by `benchmarks/solver_consistency.py`. It records numerical
consistency without defining a universal pass threshold. Timing, memory, rank selection, ordinary
PLS, software metadata, environment metadata, figures, and generic orchestration do not belong in
this table. A future runtime benchmark requires a separate hardware and measurement question.

## Output and reproducibility policy

Each benchmark owns its own script, scenario constants, tests, and CSV header. Prefer readable
ordinary Python over generic orchestration. A small configuration file is acceptable only when it
makes that benchmark clearer to both humans and machines.

Generated results remain under `benchmarks/results/`, are ignored by Git, and are excluded from
repository snapshots unless a later decision freezes a small package-validation fixture with a
stated meaning, tolerance, and update procedure.
The source commit, benchmark script, explicit seed, and package dependencies provide reproducibility;
routine result tables need not repeat software versions or execution controls when those fields do
not answer the benchmark question.

Tests may verify deterministic generation, finite metrics, metric domains, exact CSV headers, and
repeatability of scientific values. They must not freeze a claim that Pi-PLS outperforms PLS or that
rank selection is exact until a separate calibrated acceptance decision exists.

## Implementation sequence

The implemented focused sequence is:

1. Fixed-structure recovery;
2. Rank selection;
3. Predictor-nuisance comparison against ordinary PLS;
4. Exact-versus-randomized SVD consistency.

Do not add real-data benchmark copies of public examples. A new benchmark requires a distinct
package-level validation question that is not already answered by an example or a focused unit,
integration, or artifact-contract test.
