# Decision 0027: Synthetic package-benchmark contract
> Status: superseded in its universal manifest/runner/schema consequences by Decision 0030.

Historical status: accepted and implemented before Decision 0030.

## Context

After separating the long-lived package repository from paper reproduction, `pipls` needed a
validation benchmark strategy that serves programming users without importing manuscript-scale
simulations, figures, or paper-only comparators. The deterministic synthetic generator exposes
shared, predictor-specific, and response-specific latent truth, making it suitable for controlled
package validation.

The benchmark design also had to preserve model-internal fold-local centering and scaling, avoid
premature performance claims, and remain small enough for ordinary software maintenance.

## Decision

- Introduce a versioned synthetic benchmark contract under `benchmarks/` before implementing
  runners or freezing broad numerical results.
- Use `make_pipls_train_test` so train and test share the latent model but use independent sample
  realizations.
- Cover shared-only, predictor-nuisance, response-nuisance, low-sample, weak-signal,
  heterogeneous-scale, and high-dimensional solver scenarios.
- Separate fixed oracle validation from Pi-PLS path-selection validation.
- Use ordinary `PLSRegression` as the only external comparator in version 1.
- Exclude OLS, CCA, publication grids, and figure generation from the package suite.
- Define CI, standard, and opt-in performance tiers with deterministic seed sets and no
  cross-machine timing gates.
- Record prediction, selection, subspace, numerical-consistency, timing, and optional memory
  metrics in versioned JSON Lines records.
- Freeze only generator repeatability, deterministic-fit repeatability, finite metric requirements,
  and subspace metric bounds. Calibrate any predictive, selection, solver-agreement, or performance
  threshold before making it a CI gate.

## Consequences

The benchmark manifest and result schema are executable package contracts and may be structurally
tested even though ordinary roadmap prose and documentary dataset metadata remain living content.
Generated results are ignored by default. The next increment may implement the small CI runner and
tests against this contract without changing estimator behavior or designing block-aware scaling.