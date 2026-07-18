# Decision 0039: fixed estimator and path-search boundary

## Status

Accepted and partially implemented. The fixed-estimator and sole-path-ownership steps are
complete; private-code consolidation, example alignment, and final audit remain staged follow-up
work.

## Context

Before implementation, `PiPLSRegression` supported both fixed-rank fitting and internally
cross-validated predictor-rank selection. `PiPLSPathCV` separately performed the intended triangular search over
response component count and predictor rank. This duplicates model-selection responsibility and
makes an apparently ordinary estimator fit perform hidden cross-validation.

The standard package workflow needs a bounded triangular scan. Users should not have to construct
an explicit rectangular `GridSearchCV` grid for `(n_components, predictor_rank)`, and wrapping an
internally cross-validating estimator in `GridSearchCV` would create nested folds. The package
therefore needs a direct estimator for one model and a dedicated path meta-estimator for the
standard selection procedure.

## Decision

`PiPLSRegression` will fit one fixed pair $(h,r_\pi)$.

- `predictor_rank` will be an explicit positive integer rather than a search mode or a rule-derived
  ceiling.
- `fit()` will perform no cross-validation or parameter selection.
- Cross-validation controls, scoring controls, OOF reporting, and search-result attributes will not
  belong to the fixed estimator.
- A direct fixed fit will emit `StatisticalSupportWarning` when
  $n/r_\pi < 4$. Algebraically infeasible ranks remain errors.

`PiPLSPathCV` will be the standard package interface for model selection.

- It will evaluate the admissible triangular surface
  $1 \leq h \leq r_\pi \leq r_{\pi,\max}$.
- Its default statistical-support ceiling will continue to use
  `samples_per_predictor_rank=5` and the total number of observations supplied to `fit()`.
- Centered training-fold dimensions will remain hard feasibility caps.
- It will fit fixed-rank `PiPLSRegression` clones for candidate evaluation and will conditionally
  select one numeric predictor rank for every component count.
- It will suppress only the expected `StatisticalSupportWarning` from its controlled internal
  candidate fits. Other warnings and errors remain visible.

Fixed `PiPLSRegression` instances will remain compatible with ordinary scikit-learn
meta-estimators when users supply explicit valid parameter pairs. The repository will not present
`GridSearchCV` as the recommended Pi-PLS selection workflow; examples and guides will use
`PiPLSPathCV`.

## Transition

This decision is implemented incrementally:

1. simplify `PiPLSRegression` to fixed-model fitting;
2. make `PiPLSPathCV` the sole triangular-selection interface;
3. remove obsolete shared machinery and duplicated public results;
4. align examples and guides with the final two-stage workflow;
5. perform a final API and minimality audit.

Steps 1 and 2 are implemented. `PiPLSRegression` is fixed-rank only, and `PiPLSPathCV` now owns the
complete package selection lifecycle: feature probes, candidate folds, conditional path results,
optional OOF fits, and selected full-data refit. The path supplies the warning-suppression policy to
the private fold engine, which otherwise propagates warnings normally. Remaining transition work
is private-code cleanup, example alignment, and final audit.

## Consequences

- The fixed estimator will more closely resemble scikit-learn's direct regression estimators.
- The standard Pi-PLS selection path remains bounded, adaptive, and package-owned.
- Ordinary path analysis uses one CV layer; nested CV occurs only when a user deliberately places a
  selection procedure inside an external assessment procedure.
- The $c=5$ path ceiling and the $c=4$ direct-fit warning have distinct purposes.
- Decision 0032's full-sample support convention remains in force.
- The selection-responsibility portions of Decisions 0003, 0009, 0011, 0012, and 0031 are
  superseded when the staged implementation is complete.
