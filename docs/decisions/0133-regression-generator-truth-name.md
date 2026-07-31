# Decision 0133: regression-generator truth naming

## Status

Accepted and implemented.

## Context

The package exposes two distinct synthetic-data families. `make_pipls_regression()` and
`make_pipls_train_test()` implement the configurable regression generator, while
`make_pipls_latent_geometry()` implements the companion-manuscript Gaussian latent geometry.

The pre-release result class `PiPLSSyntheticTruth` belonged only to the configurable regression
family. Its generic name suggested that it also described the latent-geometry generator, which
instead returns `PiPLSLatentGeometryTruth`. The corresponding private names also used generic
`Synthetic` and `dataset block` terminology even though they implement only the configurable
regression family.

The package remains unreleased at version `0.0.0`, so no published compatibility commitment
requires retaining the former public name.

## Decision

Rename `PiPLSSyntheticTruth` to `PiPLSRegressionTruth`.

Use regression-specific private implementation names for the configurable generator:

- `_RegressionLoadings`;
- `_RegressionGeneratorConfig`;
- `_validated_regression_config()`;
- `_draw_regression_loadings()`;
- `_draw_regression_block()`;
- `_latent_signal()`;
- `_validate_dataset_truth()`.

Keep the public generator names `make_pipls_regression()`, `make_pipls_train_test()`, and
`make_pipls_latent_geometry()` unchanged. Keep `PiPLSLatentGeometryTruth` unchanged.

Use the new truth-class name throughout implementation, tests, public documentation, generated
source documentation, and active guide-layer contracts. Do not add an alias, fallback import, or
serialization migration for the former unreleased name.

Historical decision records remain unchanged. This decision supersedes Decisions 0015, 0088, and
0119 only where they require the former configurable-generator truth-class spelling. Their dataset,
result-safety, and latent-geometry contracts otherwise remain in force.

## Consequences

- The two synthetic generator families have distinct and accurate public truth-record names.
- Internal helper names identify the configurable regression generator rather than synthetic data
  in general.
- Generated arrays, validation, deterministic random streams, train/test sharing, and truth-field
  contents remain unchanged.
- Existing pre-release code importing `PiPLSSyntheticTruth` must import `PiPLSRegressionTruth`.
- No duplicate compatibility class enlarges the public dataset surface.
