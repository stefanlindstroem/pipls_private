# Decision 0119: manuscript latent-geometry generator

## Status

Accepted and implemented.

## Context

The package already provides `make_pipls_regression()` as a broad, configurable synthetic-data
facility. That generator centers and standardizes latent score columns, uses orthonormal loading
directions, and supports configurable strengths, distributions, and observed-variable scales. It is
a useful package capability, but it does not reproduce the
Gaussian latent-space data-generating model used in the companion manuscript.

The package must be capable of generating data from the manuscript model without changing its
practical real-data workflow, estimator defaults, path selection, existing examples, or existing
synthetic benchmark contracts.

## Decision

Add a separate public function, `make_synthetic_data()`, implementing

\[
X = \Lambda_p L_p + \Lambda_s L_{sp} + \varepsilon_X,
\qquad
Y = \Lambda_s L_{sr} + \Lambda_r L_r + \varepsilon_Y.
\]

The three latent-score matrices and four loading matrices contain independent standard-normal
entries. Predictor and response noise contain independent Gaussian entries with caller-specified
standard deviations. The generator applies no score centering, score standardization, loading
orthonormalization, latent-strength scaling, or observed-variable scaling.

The draw order is public and deterministic for a fixed unsigned 32-bit `random_state`:

1. predictor-specific scores `Lambda_p`;
2. shared scores `Lambda_s`;
3. response-specific scores `Lambda_r`;
4. predictor-specific loadings `L_p`;
5. shared predictor loadings `L_sp`;
6. shared response loadings `L_sr`;
7. response-specific loadings `L_r`;
8. predictor noise;
9. response noise.

Return an ordinary `PiPLSDataset` whose `truth` is a new immutable
`SyntheticDataTruth`. Loading matrices retain manuscript orientation: latent dimensions are
rows and observed variables are columns. The truth record validates both signal equations and
reconstructs through the same validation path after pickling.

The existing configurable generators remain unchanged. No estimator, search, validation,
real-data, example, benchmark, or model-selection behavior changes in this decision.

## Consequences

- The package can generate the companion manuscript's synthetic distribution directly.
- Existing package synthetic workflows remain numerically and behaviorally stable.
- Exact reproduction of a manuscript figure or table still requires its parameter grid, seeds,
  resampling protocol, and analysis settings.
- Theory and terminology alignment remain separate documentation increments.
