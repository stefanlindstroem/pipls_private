# Lightweight validation benchmarks

`pipls` uses small focused benchmarks to make selected package behavior understandable across
releases. Synthetic questions remain primary; separately reviewed real-data smoke checks exercise
ordinary user workflows. These assets are not scientific-paper reproduction studies.

## Focused design

Each benchmark answers one question and writes one small CSV file containing only the columns needed
for that question. The package does not use a universal experiment table with columns for every
method, metric, software version, execution control, and timing measurement.

## Fixed-structure recovery

The implemented fixed-structure benchmark asks:

> When the true shared dimension and predictor-signal rank are supplied, does fixed Pi-PLS recover
> the intended latent subspaces and predict independent responses?

Run it from the repository root after installing the package:

```bash
python benchmarks/fixed_structure_recovery.py
```

The script uses `make_pipls_train_test` with 160 training samples, 160 test samples, 24 predictors,
six responses, two shared directions, no response-specific directions, shared strengths `(2.5,
1.5)`, block noise `(0.2, 0.2)`, and seeds 1729, 2718, and 3141. The scenarios are:

- `shared_only`, with no predictor-specific directions;
- `predictor_specific_nuisance`, with four predictor-only directions of strengths `(3.0, 2.5,
  2.0, 1.5)`.

For each generated problem, `PiPLSRegression` uses `scale=True`, exact predictor SVD, and fixed oracle
ranks: `n_components` equals the declared shared rank, and `predictor_rank` equals the complete
declared predictor-signal rank. No cross-validation is performed.

The output is `benchmarks/results/fixed_structure_recovery.csv` with exactly these columns:

```text
scenario
seed
test_mse
predictor_shared_capture
predictor_signal_capture
response_shared_capture
```

`test_mse` is the arithmetic mean of squared test residuals over all samples and responses in the
original generated response units.

For a true basis $A$ and estimated basis $B$, let $Q_A$ and $Q_B$ be orthonormal bases for their
column spaces. Every capture metric is

egin{equation}
\mathrm{capture}(A, B) = \frac{\lVert Q_A^{\mathsf{T}} Q_B \rVert_{\mathrm{F}}^2}{\dim[\mathrm{col}(A)]}.
\end{equation}

It is the mean squared canonical correlation and lies in $[0, 1]$. The three applications are:

- `predictor_shared_capture`: true predictor-shared loadings against fitted $P$;
- `predictor_signal_capture`: concatenated true shared and predictor-specific loadings against
  fitted $\Pi$;
- `response_shared_capture`: true response-shared loadings against fitted $Q$.

The generator loadings precede observed-variable scaling, whereas the fitted bases use model
standardization. Before comparison, predictor truth loadings are multiplied row-wise by
`truth.feature_scale / model.x_scale_`, and response truth loadings by
`truth.target_scale / model.y_scale_`. Thus each comparison uses the fitted model coordinates, and
all learned scales come only from the benchmark training block.

The benchmark does not perform rank selection, compare against ordinary PLS, compare SVD solvers,
measure runtime, record software or environment metadata, or create figures.

## Rank selection

The implemented rank-selection benchmark asks:

> When the generator declares the shared dimension and complete predictor-signal rank, which ranks
> does adaptive `PiPLSPathCV(search_method="auto")` select for prediction?

Run it from the repository root after installing the package:

```bash
python benchmarks/rank_selection.py
```

The script uses `make_pipls_train_test` with 160 training samples, 160 test samples, 24 predictors,
six responses, no response-specific directions, block noise `(0.2, 0.2)`, and seeds 1729, 2718,
and 3141. The scenarios are:

- `one_shared`, with one shared direction of strength `(2.5,)`;
- `two_shared`, with two shared directions of strengths `(2.5, 1.5)`;
- `two_shared_with_predictor_nuisance`, retaining the two shared directions and adding four
  predictor-only directions of strengths `(3.0, 2.5, 2.0, 1.5)`.

The search uses `PiPLSRegression(scale=True, svd_solver="full")` inside five-fold
`PiPLSPathCV(search_method="auto")`, with a benchmark-specific rule-based predictor-rank bound
and `samples_per_predictor_rank=10.0`. Every candidate learns centering and scaling from its training
fold only. The selected estimator is then refitted on the complete generated training block.

The output is `benchmarks/results/rank_selection.csv` with exactly these columns:

```text
scenario
seed
true_n_components
selected_n_components
true_predictor_rank
selected_predictor_rank
test_mse
```

`true_n_components` is the generator-declared shared rank.
`true_predictor_rank` is the sum of the generator-declared shared and predictor-specific ranks.
The selected fields are the global path-search choice. `test_mse` is the arithmetic mean of squared
residuals over the independent test samples and responses in original response units.

The declared ranks describe the data-generating structure. The selected ranks optimize a finite
cross-validation estimate of predictive loss and need not equal the declared ranks. The benchmark
therefore records behavior without defining exact recovery, a maximum rank error, or a predictive
pass threshold.

It does not compare against fixed oracle Pi-PLS, ordinary PLS, exhaustive path search, or another
SVD solver. It also excludes subspace metrics, timings, candidate counts, software or environment
metadata, and figures.

## Predictor-nuisance comparison with ordinary PLS

The implemented paired benchmark asks:

> When predictor-specific variation increases, how does fixed Pi-PLS prediction compare with
> ordinary fixed-component PLS on the same generated train/test problem?

Run it from the repository root after installing the package:

```bash
python benchmarks/predictor_nuisance_comparison.py
```

The script uses `make_pipls_train_test` with 160 training samples, 160 test samples, 24 predictors,
six responses, two shared directions of strengths `(2.5, 1.5)`, no response-specific directions,
noise `(0.2, 0.2)`, and seeds 1729, 2718, and 3141. The scenarios are:

- `shared_only`, with no predictor-specific directions;
- `moderate_predictor_nuisance`, with four predictor-only directions of strengths `(1.5, 1.25,
  1.0, 0.75)`;
- `strong_predictor_nuisance`, with the same four directions strengthened to `(3.0, 2.5, 2.0,
  1.5)`.

Both methods use `scale=True` and `n_components` equal to the generator-declared shared rank. Fixed
Pi-PLS uses exact predictor SVD and `predictor_rank` equal to the complete declared predictor-signal
rank. Each model learns its own centering and scaling statistics from the generated training block.
No cross-validation or preprocessing fit outside the model is used.

The output is `benchmarks/results/predictor_nuisance_comparison.csv` with exactly these columns:

```text
scenario
seed
pipls_test_mse
pls_test_mse
pipls_minus_pls_mse
```

Both MSE values are arithmetic means of squared residuals over the independent test samples and
responses in original response units. The paired difference is
`pipls_test_mse - pls_test_mse`: negative values favor Pi-PLS for that generated problem, while
positive values favor ordinary PLS.

The benchmark is descriptive. It does not define a superiority threshold, aggregate across seeds,
select ranks, compare SVD solvers, measure runtime, record software or environment metadata, or
produce figures. OLS and CCA remain outside the package benchmark.

## Full-versus-randomized solver consistency

The implemented solver-consistency benchmark asks:

> For a fixed high-dimensional Pi-PLS model, how different are predictions and coefficients when
> only the predictor decomposition changes from full to randomized SVD?

Run it from the repository root after installing the package:

```bash
python benchmarks/solver_consistency.py
```

The paired models use `scale=True`, the generator-declared shared rank, and the complete declared
predictor-signal rank. One fit uses `svd_solver="full"` and the other uses
`svd_solver="randomized"`. Generated data, model ranks, and seed are identical within each pair.
The three scenarios keep the training predictor matrix at 36,864 entries while changing its
geometry:

- `wide_n96_p384`, with 96 training samples and 384 predictors;
- `square_n192_p192`, with 192 training samples and 192 predictors;
- `tall_n384_p96`, with 384 training samples and 96 predictors.

All scenarios use 96 test samples, eight responses, three shared directions of strengths `(3.0,
2.0, 1.5)`, five predictor-specific directions of strengths `(2.5, 2.0, 1.5, 1.0, 0.75)`, no
response-specific directions, noise `(0.3, 0.2)`, and seeds 1729, 2718, and 3141.

The output is `benchmarks/results/solver_consistency.csv` with exactly these columns:

```text
scenario
seed
prediction_relative_difference
coefficient_relative_difference
```

With the full-SVD result as reference, the metrics are

\begin{equation}
 d_{\mathrm{prediction}}
 =
 \frac{\lVert \widehat{Y}_{\mathrm{randomized}} - \widehat{Y}_{\mathrm{full}} \rVert_{\mathrm{F}}}
 {\max(\lVert \widehat{Y}_{\mathrm{full}} \rVert_{\mathrm{F}}, \epsilon)},
\end{equation}

and

\begin{equation}
 d_{\mathrm{coefficient}}
 =
 \frac{\lVert B_{\mathrm{randomized}} - B_{\mathrm{full}} \rVert_{\mathrm{F}}}
 {\max(\lVert B_{\mathrm{full}} \rVert_{\mathrm{F}}, \epsilon)},
\end{equation}

where $\epsilon$ is machine epsilon. Predictions and coefficients are in original response units.
The benchmark is descriptive: it does not define a universal pass threshold, compare prediction
accuracy, select ranks, measure runtime or memory, or record software and environment metadata.

## Pulp component-path smoke check

The first real-data smoke check asks:

> Can an ordinary programming user read the transparent Pulp tables and obtain one explicit
> predictor-rank and CV-MSE result for every candidate component count?

Run it from the repository root after installing the data dependencies:

```bash
python benchmarks/pulp_path_smoke.py
```

The script reads `datasets/pulp/X.csv` and `Y.csv` directly with pandas and evaluates:

```python
search = PiPLSPathCV(
    n_components_values=[1, 2, 3, 4],
    refit=False,
    n_jobs=1,
).fit(X, Y)
```

The output is `benchmarks/results/pulp_path_smoke.csv` with four ordered rows and exactly:

```text
n_components
predictor_rank
predictor_rank_policy
response_standardized_cv_mse_mean
response_standardized_cv_mse_fold_sd
n_splits
```

Predictor rank is selected conditionally for each component count and is always recorded as a
numeric value. `predictor_rank_policy` states how the rank was obtained. The fold SD is the
population standard deviation of the fold-specific response-standardized MSE values. It is
descriptive variation across overlapping folds, not a confidence interval.

The benchmark does not use `best_params_` to declare a final model, does not refit a globally
selected pair, and does not generate a plot. The corresponding public example writes the same path
schema to CSV, generates a PDF from that CSV, and then performs a separate explicit fixed-model
fit.

## Sugarcane high-dimensional component-path smoke check

The second real-data smoke check asks:

> Does the same transparent component-path workflow operate when the predictor matrix is
> high-dimensional with $p \gg n$?

Run it with:

```bash
python benchmarks/sugarcane_path_smoke.py
```

The script reads the 57-row, 1,721-predictor Sugarcane `X.csv` and four-response `Y.csv` tables
directly with pandas and uses the same default `PiPLSPathCV(refit=False)` call over component counts
1 through 4. It writes `benchmarks/results/sugarcane_path_smoke.csv` with the same six columns as the
Pulp path.

The check records one numeric predictor rank, policy, mean CV-MSE, fold SD, and split count per
component count. It does not set an explicit predictor-rank ceiling, force randomized SVD, select
spectral preprocessing, choose the final model, or report timing.

## Tobacco full-SVD component-path smoke check

The third real-data smoke check asks:

> Can a package user evaluate the Tobacco component path with adaptive predictor-rank scanning
> while using an explicit exact predictor decomposition?

Run it with:

```bash
python benchmarks/tobacco_path_smoke.py
```

The script reads the 347-row, 1,557-predictor Tobacco `X.csv` and 13-response `Y.csv` tables
directly with pandas. It uses:

```python
search = PiPLSPathCV(
    estimator=PiPLSRegression(svd_solver="full"),
    n_components_values=range(1, 9),
    search_method="auto",
    refit=False,
    n_jobs=1,
).fit(X, Y)
```

The output is `benchmarks/results/tobacco_path_smoke.csv`, with the same six columns as the Pulp
and Sugarcane component paths. The numeric predictor rank is recorded for every component count.
The smoke check remains Pi-PLS-only; standard PLS comparison CSVs and PDFs belong to the public
examples.

The benchmark does not select a final component count, refit a final model, introduce spectral
preprocessing, report timing or memory, or define a performance threshold. Randomized predictor
SVD remains covered by the focused full-versus-randomized solver-consistency benchmark.

## Benchmark sequence status

All four planned focused synthetic benchmarks and the Pulp, Sugarcane, and Tobacco
component-path smoke checks are implemented. Each real-data check retains its own question and
computational boundary rather than being appended to a universal table.

## Interpretation boundary

Synthetic train and test blocks are generated independently from shared latent parameters. Models
learn centering and optional scaling only from their training data; any future cross-validation
benchmark must learn those statistics independently within each fold.

Ordinary PLS is the nearest package-user comparator where a comparison is the stated question. OLS,
CCA, publication-scale simulations, figures, and scientific superiority claims remain outside this
repository.

Generated CSV files live under `benchmarks/results/`, are ignored by Git, and are excluded from
repository snapshots. Timings and software versions are not included automatically; they belong
only in a separately designed benchmark whose question concerns runtime or compatibility.
