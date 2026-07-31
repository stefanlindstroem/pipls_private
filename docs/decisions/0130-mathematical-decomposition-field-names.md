# Decision 0130: mathematical decomposition field names

## Status

Accepted and implemented.

## Context

The canonical Pi-PLS terminology established by Decisions 0121 and 0122 calls the columns of
$\mathbf{P}$ and $\mathbf{Q}$ predictor and response directions. The immutable
`PiPLSDecomposition` nevertheless retained the pre-release field names `predictor_rotations` and
`response_rotations` for anticipated compatibility. This left the public code vocabulary at odds
with the mathematical vocabulary and required every user-facing explanation to translate between
names for the same objects.

The package remains unreleased at version `0.0.0`. No published compatibility commitment requires
retaining those field names or aliases for them.

## Decision

Rename the public immutable decomposition fields:

- `predictor_rotations` to `predictor_directions`;
- `response_rotations` to `response_directions`.

Use the new names throughout implementation, tests, examples, generated docstrings, public
documentation, and active guide-layer contracts. Do not add compatibility properties, constructor
aliases, `__getattr__` fallbacks, or serialization migration code for the former names.

Retain `PiPLSRegression.x_rotations_` and `y_rotations_`. Those are standard PLS-family fitted
attribute names used for scikit-learn interoperability. They reference the same read-only arrays as
`decomposition_.predictor_directions` and `decomposition_.response_directions`; mathematical prose
continues to call the arrays directions. The separate reconstruction quantities remain
`x_loadings_` and `y_loadings_`.

Historical decision records remain unchanged. This decision supersedes Decisions 0121 and 0122
only where they require the pre-release decomposition field names containing `rotations`. Their
remaining terminology and documentation contracts remain in force.

## Consequences

- Public code and mathematical documentation use one name for each Pi-PLS direction array.
- Direct `PiPLSDecomposition` construction uses the new keyword names.
- Existing pre-release pickles or code using the former field names are intentionally unsupported.
- Estimator fitting, numerical values, array identity, immutability, shapes, scikit-learn fitted
  attributes, and standardized regression maps are unchanged.
