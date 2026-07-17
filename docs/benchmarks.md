# Lightweight validation benchmarks

`pipls` uses small synthetic benchmarks to make selected package behavior understandable across
releases. They are designed for programming users, not for reproducing a scientific paper.

## Focused design

Each benchmark answers one question and writes one small CSV file containing only the columns needed
for that question. The package does not use a universal experiment table with columns for every
method, metric, software version, execution control, and timing measurement.

The planned benchmark sequence is:

### Fixed-structure recovery

Does fixed Pi-PLS recover known predictor and response subspaces and predict independent synthetic
responses when the true model dimensions are supplied?

The table will contain scenario, seed, test MSE, and three subspace-capture metrics.

### Rank selection

Does adaptive `PiPLSPathCV` choose reasonable shared and predictor ranks when the generator declares
the truth?

The table will contain scenario, seed, true ranks, selected ranks, and test MSE.

### Predictor-nuisance comparison

As predictor-specific nuisance variation increases, how does fixed Pi-PLS prediction compare with
ordinary fixed-component `PLSRegression`?

The table will contain paired Pi-PLS and PLS test MSE values and their difference.

### Solver consistency

For a fixed high-dimensional model, how closely do full and randomized predictor SVD agree?

The table will contain only prediction and coefficient relative differences.

## Interpretation boundary

Synthetic train and test blocks are generated independently from shared latent parameters. Models
learn centering and optional scaling only from their training data; cross-validation learns those
statistics independently within each fold.

Ordinary PLS is the nearest package-user comparator. OLS, CCA, publication-scale simulations,
figures, and scientific superiority claims remain outside this repository.

Generated CSV files live under `benchmarks/results/` and are ignored by Git. Timings and software
versions are not included automatically; they belong only in a separately designed benchmark whose
question concerns runtime or compatibility.
