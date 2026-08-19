# Decision 0039: fixed estimator and path-search boundary

## Status

Accepted and fully implemented for estimator/search ownership. Decision 0154 supersedes only the
search support ceiling and default coverage policy recorded here; that migration is implemented.
Decision 0155 preserves this ownership boundary and assigns response-subspace policy to the fixed
estimator.

## Context

Before implementation, `PiPLSRegression` supported both fixed-rank fitting and internally
cross-validated predictor-rank selection. `PiPLSSearchCV` separately performed the intended triangular search over
response component count and predictor rank. This duplicates model-selection responsibility and
makes an apparently ordinary estimator fit perform hidden cross-validation.

The standard package workflow needs a bounded triangular scan. Users should not have to construct
an explicit rectangular `GridSearchCV` grid for `(n_components, predictor_rank)`, and wrapping an
internally cross-validating estimator in `GridSearchCV` would create nested folds. The package
therefore needs a direct estimator for one model and a dedicated search meta-estimator for the
standard selection procedure.

## Decision

`PiPLSRegression` will fit one fixed pair $(h,r_\pi)$.

- `predictor_rank` will be an explicit positive integer rather than a search mode or a rule-derived
  ceiling.
- `fit()` will perform no cross-validation or parameter selection.
- Cross-validation controls, scoring controls, OOF reporting, and search-result attributes will not
  belong to the fixed estimator.
- A direct fixed fit will emit `StatisticalSupportWarning` when
  $n/r_\pi < 3$. Algebraically infeasible ranks remain errors.

`PiPLSSearchCV` will be the standard package interface for model selection.

- It will evaluate the admissible triangular surface
  $1 \leq h \leq r_\pi \leq r_{\pi,\max}$.
- Its ordinary automatic rank domain is bounded only by hard fold feasibility and any explicit
  integer `max_predictor_rank`; `samples_per_predictor_rank` belongs only to the explicit EPV policy.
- Centered training-fold dimensions and verified numerical rank remain hard feasibility caps.
- It will fit fixed-rank `PiPLSRegression` clones for candidate evaluation and will conditionally
  select one numeric predictor rank for every component count.
- It will suppress only the expected `StatisticalSupportWarning` from its controlled internal
  candidate fits. Other warnings and errors remain visible.

Fixed `PiPLSRegression` instances will remain compatible with ordinary scikit-learn
meta-estimators when users supply explicit valid parameter pairs. The repository will not present
`GridSearchCV` as the recommended Pi-PLS selection workflow; examples and guides will use
`PiPLSSearchCV`.

## Relationship to Decision 0155

Decision 0155 adds a fixed-estimator response-subspace configuration without adding a model-selection
dimension. `PiPLSRegression` owns `response_subspace`, while `PiPLSSearchCV` preserves that setting
when cloning the estimator template for candidate evaluation, OOF work, and final refitting. Search
continues to optimize only component count $h$ and predictor rank $r_\pi$.

Automatic comparison of the cross-covariance and least-squares response-subspace policies is
therefore outside `PiPLSSearchCV`. A controlled comparison uses separate estimator templates and,
when appropriate, the same materialized CV splits.

## Transition

This decision is implemented incrementally:

1. simplify `PiPLSRegression` to fixed-model fitting;
2. make `PiPLSSearchCV` the sole triangular-selection interface;
3. remove obsolete shared machinery and duplicated private results;
4. align examples and guides with the final two-stage workflow;
5. perform a final API and minimality audit.

All five steps are implemented. `PiPLSRegression` is fixed-rank only, and `PiPLSSearchCV` owns the
complete package selection lifecycle: feature probes, candidate folds, conditional path results,
optional OOF fits, and selected full-data refit. The path supplies the warning-suppression policy to
the private fold engine, which otherwise propagates warnings normally. Obsolete solver tracing,
unused rank-grid construction, duplicate candidate metadata, and OOF rescoring have been removed.
Examples compose their PLS comparison and CSV-to-PDF helpers as ordinary imported functions, retain
direct pandas I/O, and fit the final fixed pair from the chosen canonical CSV row.

The final audit removed the redundant public `pipls_param_prefix` constructor control because the
supported pipeline form already requires a unique terminal `PiPLSRegression` step. The terminal
parameter prefix is now inferred. A focused interoperability test confirms that fixed explicit
rank pairs remain usable with ordinary `GridSearchCV`, while repository examples continue to
recommend `PiPLSSearchCV`.

The implemented public surface additionally uses the explicit `n_components_values="all"`
sentinel, conventional random-state forms, a public callable default scorer, and one canonical
`decomposition_` location for Pi-PLS-specific fitted output. Later pre-release cleanup is summarized
in [history.md](history.md); the fixed-estimator/search boundary remains canonical here.

## Consequences

- The fixed estimator resembles scikit-learn's direct regression estimators.
- The standard Pi-PLS selection path remains package-owned; Decision 0154 makes exhaustive
  full-feasible coverage the default and retains adaptive coverage as an explicit option.
- Ordinary path analysis uses one CV layer; nested CV occurs only when a user deliberately places a
  selection procedure inside an external assessment procedure.
- The explicit EPV policy and the $c=3$ direct-fit support warning have distinct purposes; the
  default search domain has no statistical $n/c$ ceiling under Decision 0154.
- Decision 0032's full-sample support convention remains in force.
- Earlier constructor-owned and shared-engine selection arrangements are superseded by this
  implemented boundary.
- Supported pipelines require no public parameter-prefix control because the terminal Pi-PLS step
  is unique and inferred.
