# Decision 0108: Tobacco one-standard-error workflow

## Status

Accepted.

## Context

Decision 0107 introduced exact stored-value recommendation methods on `PiPLSComponentPath` and kept
them out of numbered examples while their reference contract was established. The Tobacco workflow
continued to select eight components through an unexplained literal even though it already evaluated
the complete component path and displayed fold-based standard-error bars.

Example 07 is the most appropriate maintained workflow for one concrete application of the
one-standard-error rule. It is an advanced complete real-data analysis rather than an onboarding
tutorial, and its current fixed component count has no visible empirical justification.

## Decision

`examples/07_tobacco_real_data.py` calls `component_path_.one_standard_error_result()` and uses the
returned `n_components` and aligned conditionally selected `predictor_rank` for the final fixed
`PiPLSRegression`. It does not call `minimum_cv_mse_result()` or reproduce the threshold calculation
manually.

The component-path figure labels the selected diamond as the 1-SE recommendation. The later
inspection code derives up to four displayed component indices from the recommended component count
and handles a possible one-component recommendation without invalid score indexing.

The path search itself is unchanged. `PiPLSSearchCV` does not automatically apply the rule, predictor
rank is not selected again, and no other numbered example, tutorial, renderer, README section, or
documentation-home workflow invokes either recommendation method. The path-analysis guide and
example catalogue identify example 07 as the maintained application.

## Consequences

The Tobacco final-model size now has a reproducible path-based rationale instead of a literal
component count. Programming users can inspect one complete use of the recommendation method without
making it part of the introductory documentation route. The selected component count may change if
the evaluated data, cross-validation protocol, scoring contract, or numerical path results change;
that dependency is intentional and remains visible in the example source.
