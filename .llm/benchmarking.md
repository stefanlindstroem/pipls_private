# Benchmarking contract

## Purpose

Package benchmarks protect the long-lived `pipls` software product. They are smaller and more
stable than publication experiments and answer questions relevant to programming users: prediction,
selection behavior, recovery of known structure, numerical consistency, and representative
resource use.

This file is normative for benchmark design. The machine-readable suite contract is
`benchmarks/manifests/synthetic-v1.yaml`; generated record structure is defined by
`benchmarks/schema/result-v1.schema.json`.

## Scope boundary

The package benchmark layer may contain:

- controlled deterministic synthetic scenarios;
- ordinary PLS as the nearest practical comparator;
- small real-data smoke checks after the synthetic layer is established;
- versioned manifests, runners, schemas, and narrowly frozen package-validation fixtures;
- opt-in timing and memory measurements.

It must not contain:

- complete publication simulation grids;
- manuscript figures or tables;
- paper-only OLS or CCA comparison programs;
- cached publication outputs;
- claims of scientific superiority inferred from a small package fixture.

OLS or CCA may enter only through a separate decision when a narrowly defined package-level identity
or limiting case cannot be protected otherwise.

## Data generation and leakage boundary

Use `pipls.datasets.make_pipls_train_test`. The generated train and test blocks share latent
loadings, strengths, and observed-variable scales, while their scores and noise realizations are
independent. A benchmark seed determines the complete generated problem.

Do not center, scale, or otherwise learn a transform before model fitting. Pi-PLS and ordinary PLS
fit their own training means and scales. Inner cross-validation must clone and fit the complete
candidate within every training fold, then refit the selected candidate on the complete generated
training block. Future block-aware scaling variants must obey the same boundary but are not designed
by this benchmark contract.

## Benchmark tracks

### Fixed-parameter comparison

Use synthetic truth to set:

- `n_components = n_shared` for Pi-PLS and ordinary PLS;
- `predictor_rank = n_shared + n_predictor_specific` for fixed Pi-PLS.

This oracle track isolates model and numerical behavior. It is not a real-data tuning prescription.

### Pi-PLS selection validation

Use `PiPLSPathCV(search_method="auto")` with the public rank rule, five shuffled K-fold splits, and a
seeded splitter. Record selected dimensions and external-test prediction. Compare selected ranks
with declared ranks descriptively; finite noisy samples and the support rule need not recover them
exactly.

### Solver consistency and resources

Compare fixed full and randomized predictor SVD only in the opt-in performance tier. Same-seed
randomized runs must be deterministic. Full-versus-randomized differences are recorded but do not
become CI gates until separately calibrated.

## Metrics

Prediction metrics:

- external-test response-standardized MSE using response scales learned from the generated training
  block;
- uniform-average external-test R².

Selection metrics:

- selected `n_components` and `predictor_rank`;
- absolute deviation from `n_shared` and from
  `n_shared + n_predictor_specific`.

Subspace metrics use projection overlap

$$
\operatorname{capture}(U, V)
= \frac{\lVert U^{\mathsf T}V\rVert_F^2}{\operatorname{rank}(U)},
$$

where `U` and `V` have orthonormal columns. Values lie in `[0, 1]` and are invariant to signs and
within-subspace rotations.

For a fitted estimator, transform raw synthetic predictor loadings `L_X` into core coordinates with
`diag(feature_scale / x_scale_) @ L_X`; transform response loadings analogously with
`diag(target_scale / y_scale_) @ L_Y`. Orthonormalize these transformed bases before comparing:

- shared predictor truth with `decomposition_.P`;
- complete predictor-signal truth with `decomposition_.Pi`;
- shared response truth with `decomposition_.Q`.

Resource metrics are wall-clock fit and prediction time. Peak resident memory is optional and must
be marked missing rather than estimated when unavailable.

## Runtime tiers

- **CI:** one seed and a small scenario subset; deterministic correctness only; no timing gate.
- **Standard:** every scientific scenario across five fixed seeds; local or scheduled validation;
  no cross-machine timing gate.
- **Performance:** high-dimensional solver scenario across three seeds; hardware metadata required
  when results are retained; no timing threshold without a hardware-specific policy.

The expected runtime values in the manifest are planning budgets, not pass/fail assertions.

## Result and tolerance policy

Write one JSON Lines record per suite, tier, scenario, seed, and method. Record package versions,
resolved parameters, metrics, status, and optional hardware details. Generated results live under
`benchmarks/results/` and are ignored by default.

Contract version 1 freezes only:

- exact same-seed synthetic array generation;
- deterministic fit repeatability at `rtol=1e-12`, `atol=1e-12`;
- finite required metrics;
- subspace-capture range with `1e-12` numerical slack.

Prediction thresholds, rank-recovery rates, Pi-PLS-versus-PLS differences, timing expectations, and
full-versus-randomized agreement are not frozen. A later fixture may freeze a value only with a
decision record that states its meaning, tolerance, supported environments, and update procedure.

## Versioning and updates

Treat manifests and result schemas as executable, versioned contracts rather than living narrative
metadata. Compatible clarifications may update documentation. Any change to scenario definitions,
seed sets, method semantics, metric formulas, or result fields requires either a new suite/schema
version or an explicit compatibility decision.

Never replace a benchmark fixture solely because a new result looks preferable. Recompute and
review after an intentional algorithm, dependency, tolerance, or platform-policy change; record why
an accepted frozen fixture changed.
