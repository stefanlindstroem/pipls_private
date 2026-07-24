# Decision 0102: path-search defaults and scorer presentation

## Context

The routine documented workflow evaluates a component path, lets the user choose a parsimonious
component count, and then fits one explicit fixed `PiPLSRegression`. `PiPLSPathCV` nevertheless
defaulted to `refit=True`, which automatically fitted the globally best evaluated pair and exposed
delegated prediction methods. Its default `scoring` value was also a function object, so generated
Python signatures displayed a process-specific memory address.

The package has not been released, so accidental defaults do not require compatibility handling.
The response-standardized scoring formula and public scorer callable are already accepted and must
not change.

## Decision

1. `PiPLSPathCV` defaults to `refit=False`.
2. The default object evaluates and records the path, global best pair, concise component path,
   predictor-rank profiles, and validation report without fitting a full-data prediction model.
3. Users request automatic full-data fitting of the globally best evaluated pair explicitly with
   `refit=True`; only then are delegated prediction, transformation, scoring, feature-name, and
   refitted-estimator surfaces available.
4. The default `scoring` parameter is the stable string
   `"neg_response_standardized_mean_squared_error"`.
5. That package-specific name resolves internally to the existing public callable
   `pipls.metrics.neg_response_standardized_mean_squared_error` and retains the same fold-local
   response-standardized MSE values, candidate ordering, and selection rule.
6. Ordinary scikit-learn scorer names, scorer callables, and `None` remain supported.
7. Repository consumers that require refitted-model behavior state `refit=True`; path-inspection
   workflows may continue to state `refit=False` explicitly when doing so improves readability.
8. No compatibility alias or legacy default is retained.

## Consequences

`PiPLSPathCV()` now represents the documented selection workflow directly. Its generated signature
is stable across processes, and users can copy the default scorer name through parameter grids,
cloning, and serialized configuration without embedding a function representation. The public
scorer function remains importable and reusable. Numerical scores, candidate feasibility,
selection tolerance, tie-breaking, OOF behavior, and explicit `refit=True` behavior are unchanged.
