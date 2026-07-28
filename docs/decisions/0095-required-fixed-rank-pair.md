# Decision 0095: require the fixed Pi-PLS rank pair

**Status:** Accepted
**Date:** 2026-07-24

## Context

`PiPLSRegression` fits one fixed pair $(h,r_\pi)$ and performs no parameter selection. Its
constructor nevertheless supplied defaults of two for both `n_components` and `predictor_rank`.
Those values had no general mathematical or statistical justification and allowed a caller to fit a
model without making the defining rank choice explicit.

`PiPLSSearchCV` still needs a valid terminal `PiPLSRegression` object when it owns the search or when a
user places the estimator in a pipeline. The selector replaces both rank parameters before its
fold-rank preflight, candidate fits, optional OOF fits, and final refit.

## Decision

1. `PiPLSRegression` requires `n_components` and `predictor_rank` as keyword-only constructor
   arguments. Neither parameter has a default or accepts a missing-value or automatic sentinel.
2. Optional preprocessing and solver controls retain their defaults.
3. `PiPLSSearchCV(estimator=None)` creates a private direct estimator template with the smallest valid
   construction seed pair `(n_components=1, predictor_rank=1)`.
4. A user-supplied pipeline terminal estimator must likewise contain a valid explicit pair. The pair
   is a construction seed only: `PiPLSSearchCV` replaces both values for every path-owned fit, so it
   does not constrain or select the evaluated path.
5. All repository consumers migrate in the same patch. No compatibility wrapper or deprecated
   constructor form is retained because the package has not been released.

## Consequences

Every direct fixed fit visibly states the complete model rank. Generated signatures no longer
suggest that `(2,2)` is a generally preferred model. Path selection remains behaviorally unchanged;
its internal or pipeline seed pair exists only to satisfy the fixed-estimator constructor contract.
