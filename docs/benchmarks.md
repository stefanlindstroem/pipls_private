# Lightweight validation benchmarks

`pipls` uses small synthetic benchmarks to make selected package behavior understandable across
releases. They are designed for programming users, not for reproducing a scientific paper.

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

## Planned separate benchmarks

The remaining benchmark sequence is:

1. adaptive rank selection;
2. predictor-specific nuisance comparison between fixed Pi-PLS and ordinary PLS;
3. full-versus-randomized predictor-SVD consistency.

Each will receive its own script and minimal CSV output in a separate patch.

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
