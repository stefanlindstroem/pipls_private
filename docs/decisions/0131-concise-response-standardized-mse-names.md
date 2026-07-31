# Decision 0131: concise response-standardized MSE names

## Status

Accepted and implemented.

## Context

Pi-PLS exposes positive and sign-reversed response-standardized mean squared error scorers. Their
pre-release callable names spelled out `mean_squared_error`, while component-path and validation
results consistently use the established abbreviation `mse`, including `cv_mse_mean`,
`cv_mse_fold_sd`, `cv_mse_standard_error`, and `split_response_standardized_mse`.

The longer callable names and default scorer string were precise but cumbersome in imports,
configuration, generated signatures, tests, and documentation. The package remains unreleased at
version `0.0.0`, so no published compatibility commitment requires retaining those spellings.

## Decision

Rename the public scorer callables:

- `response_standardized_mean_squared_error` to `response_standardized_mse`;
- `neg_response_standardized_mean_squared_error` to `neg_response_standardized_mse`.

Rename the stable default `PiPLSSearchCV.scoring` string to
`"neg_response_standardized_mse"`. Resolve that package-specific string to
`pipls.metrics.neg_response_standardized_mse`.

Use the concise names throughout implementation, tests, generated API directives, public
documentation, error messages, and active guide-layer contracts. Do not add callable aliases,
accept the former package scorer string, provide fallback lookup, or add serialization migration
for the former names.

The scoring definition, fold-local response scales, sign convention, candidate scores, ordering,
selection, and diagnostics remain unchanged. Ordinary scikit-learn scorer names, scorer callables,
and `None` remain accepted.

Historical decision records remain unchanged. This decision supersedes Decisions 0004, 0040, and
0102 only where they require the former callable names or default scorer string. Their mathematical,
scikit-learn interoperability, and path-search contracts otherwise remain in force.

## Consequences

- Public scorer names match the established MSE terminology used by result fields.
- The default search signature remains stable and becomes shorter.
- Existing pre-release imports and configurations using the former names are intentionally
  unsupported.
- Numerical scoring and model-selection behavior are unchanged.
