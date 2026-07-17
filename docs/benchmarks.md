# Lightweight validation benchmarks

`pipls` maintains small validation benchmarks for the software product. Their purpose is to make
important numerical and user-facing behavior reviewable across releases, not to reproduce any one
paper.

## Synthetic benchmark contract

The versioned contract is stored in
[`../benchmarks/manifests/synthetic-v1.yaml`](../benchmarks/manifests/synthetic-v1.yaml). It uses
`pipls.datasets.make_pipls_train_test` so training and test observations share the same latent
loadings and strengths but have independent score and noise realizations.

The named scenarios cover:

- shared-only structure;
- strong predictor-specific nuisance variation;
- response-specific variation that cannot be predicted from `X`;
- low-sample rank-bound behavior;
- weak shared signal;
- heterogeneous feature and response scales;
- high-dimensional predictor-SVD behavior.

The contract defines three runtime tiers. The CI tier is small and deterministic. The standard tier
covers all scenario families across several seeds. The performance tier is opt-in and records
solver timing, optional memory use, and full-versus-randomized consistency without imposing
cross-machine timing thresholds.

## Methods and interpretation

Fixed-parameter Pi-PLS and ordinary `PLSRegression` use the declared shared latent dimension. The
fixed Pi-PLS diagnostic also uses the declared complete predictor-signal rank. This is an oracle
validation setting: it isolates model behavior when synthetic truth is known and is not a claim
that those parameters are available for real data.

`PiPLSPathCV(search_method="auto")` is evaluated separately for package rank-selection behavior.
Its selected dimensions are compared descriptively with the declared synthetic ranks. Exact rank
recovery is not assumed for every finite noisy sample, especially when the public sample-support
rank rule deliberately limits the admissible path.

Ordinary PLS is the only external comparator in contract version 1. OLS and CCA are not part of the
package benchmark suite.

## Metrics

The contract records:

- test response-standardized mean squared error and uniform-average R²;
- selected Pi-PLS dimensions and absolute deviations from declared ranks;
- sign- and rotation-invariant capture of shared predictor, complete predictor-signal, and shared
  response subspaces;
- deterministic and full-versus-randomized numerical differences;
- fit time, prediction time, and optional peak resident memory.

Subspace capture is computed in the fitted estimator's centered/scaled coordinates. Synthetic
loading matrices are transformed by the observed-variable generator scales divided by the fitted
training scales, orthonormalized, and compared through squared projection overlap. This preserves
the current model-internal standardization contract.

## Results and tolerances

Generated records use JSON Lines and the schema at
[`../benchmarks/schema/result-v1.schema.json`](../benchmarks/schema/result-v1.schema.json). Result
files are not committed by default.

Contract version 1 freezes exact generator repeatability, strict deterministic-fit repeatability,
finite metric requirements, and the `[0, 1]` range of subspace-capture metrics. It does not freeze
predictive superiority, exact selection rates, timing limits, or full-versus-randomized agreement
thresholds. Those values require empirical calibration and a separate decision before becoming CI
gates.
