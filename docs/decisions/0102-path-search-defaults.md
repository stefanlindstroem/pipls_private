# Decision 0102: path-search defaults and scorer presentation

## Status

Accepted and implemented for selection-only path evaluation and stable scorer presentation.
Decision 0137 supersedes the former constructor-time refit and delegated fitted-model surfaces;
Decisions 0140 and 0148 define current conditioned-path selection.

## Context

The routine workflow evaluates candidate evidence, lets the user choose a parsimonious component
count, and then fits one explicit fixed `PiPLSRegression`. Earlier pre-release search behavior could
fit a global candidate automatically and delegate prediction methods from the search object. Its
default `scoring` value was also a function object, so generated Python signatures displayed a
process-specific memory address.

The response-standardized scoring formula and public scorer callable were already accepted and must
not change.

## Decision

1. `PiPLSSearchCV.fit()` evaluates and stores candidate evidence without fitting a final full-data
   model.
2. The fitted search exposes `cv_results_`, the predictor-rank-conditioned `component_path_`, and
   on-demand predictor-rank profiles. It exposes no fitted global-best attributes or delegated
   prediction methods.
3. Users construct a final full-data model explicitly with post-fit `refit(...)`, using a named
   conditioned-path rule or an evaluated component count.
4. The default `scoring` parameter is the stable string `"neg_response_standardized_mse"`.
5. That package-specific name resolves internally to the public callable
   `pipls.metrics.neg_response_standardized_mse` and retains the same fold-local
   response-standardized MSE values and configured-score orientation.
6. Ordinary scikit-learn scorer names, scorer callables, and `None` remain supported.
7. No compatibility alias for the removed constructor-time refit or obsolete scorer name is
   retained.

## Consequences

`PiPLSSearchCV()` represents path evaluation and explicit post-fit selection. Its generated
signature is stable across processes, and users can copy the default scorer name through parameter
grids, cloning, and serialized configuration without embedding a function representation. The
public scorer function remains importable and reusable. Candidate evaluation, fold-local response
scales, rank feasibility, adaptive coverage, and immutable selection provenance remain governed by
their focused decisions.
