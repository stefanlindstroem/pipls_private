# Decision 0015: dataset container and synthetic generator API

## Status

Accepted in Phase E1; simplified to the single synthetic generator.

## Context

The package needs one stable in-memory boundary for packaged datasets, synthetic validation data,
and structured experiments. The boundary must preserve names, identifiers, provenance, and
synthetic truth without making preprocessing decisions or exposing mutable arrays. Synthetic
validation and examples also need deterministic predictor-specific, shared, and response-specific
latent structure without changing NumPy's global random state.

The synthetic facility is support infrastructure for testing, validation, examples, and
reproducibility. It is not part of the Pi-PLS estimator or fitting algorithm, so the public surface
should contain no more generator machinery than those uses require.

## Decision

- Public dataset functionality lives in `pipls.datasets`, following the scikit-learn convention of
  a dedicated dataset namespace rather than expanding the package root.
- `PiPLSDataset` is the common immutable, validation-controlled container. It stores `X`, `Y`,
  feature names, target names, sample identifiers, provenance, metadata, and optional synthetic
  truth.
- `data` and `target` are scikit-learn-style aliases for `X` and `Y`.
- `Y` is normalized to a two-dimensional array, including single-response datasets.
- Required provenance keys are `source`, `license`, `citation`, and `version` when the optional
  container is used. They are not prerequisites for fitting plain user-supplied `X` and `Y`.
- All model arrays and array-valued metadata are copied and made read-only. Metadata NumPy arrays
  must have a non-object dtype; heterogeneous values use nested sequences or mappings so their
  contents can be recursively frozen. Unsupported mutable or object values are rejected.
- `make_synthetic_data()` is the single public synthetic generator. It uses a local seeded
  `numpy.random.Generator` and returns one `PiPLSDataset`.
- `SyntheticDataTruth` is the single synthetic truth record. It exposes read-only latent scores,
  loading matrices, signal matrices, and noise matrices for direct validation of the generated
  construction.
- The generator accepts predictor-specific, shared, and response-specific latent dimensions,
  predictor/response noise levels, and one deterministic unsigned 32-bit seed. Zero latent ranks
  and zero noise are valid.
- Train/test partitioning is not generator API. Generate one reproducible dataset and split rows
  explicitly when separate analysis blocks are needed.
- Real dataset loading, implicit or explicit downloading, conversion, checksums, and registry
  resolution are not part of the synthetic generator contract.

## Consequences

- Synthetic estimator tests and examples use one package-owned generator rather than parallel
  synthetic frameworks.
- Train/test demonstrations use ordinary row splitting without a separate generator abstraction.
- Real-data examples may use plain arrays or data frames and are not required to construct this
  container.
- Dataset migrations must keep their reading and matrix-construction steps explicit rather than
  adding a generic runtime loader.
