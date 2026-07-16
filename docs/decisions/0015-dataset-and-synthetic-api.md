# Decision 0015: dataset container and synthetic generator API

## Status

Accepted in Phase E1.

## Context

The package needs one stable in-memory boundary before real datasets, registry loading, and paper
reproduction are introduced. The boundary must preserve names, identifiers, provenance, and
synthetic truth without making preprocessing decisions or exposing mutable arrays. Synthetic
studies also need deterministic shared, predictor-specific, and response-specific latent
structure without changing NumPy's global random state.

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
- All model arrays and array-valued metadata are copied and made read-only. Nested metadata is
  recursively frozen and unsupported mutable/object values are rejected.
- `make_pipls_regression` uses a local seeded `numpy.random.Generator` and returns one
  `PiPLSDataset`.
- `make_pipls_train_test` returns two `PiPLSDataset` objects generated from one shared latent
  loading model but independent sample scores and noise. It performs no fitted preprocessing.
- `PiPLSSyntheticTruth` exposes read-only latent scores, loading blocks, signal/noise matrices,
  strengths, and scales. Predictor-side response-specific and response-side predictor-specific
  loading blocks are explicit zeros.
- Active latent strengths and observed-variable scales must be positive and finite. Noise may be
  zero but not negative. Zero latent ranks are valid negative-control configurations.
- Each generated sample block must contain more rows than the larger declared centered latent rank;
  degenerate blocks that cannot realize the requested rank are rejected.
- Real dataset loading, implicit or explicit downloading, conversion, checksums, and registry
  resolution are not part of E1.

## Consequences

- Synthetic estimator tests can use one package-owned generator rather than ad hoc local formulas.
- Train/test demonstrations can share a true model without fitting transformations across the
  boundary.
- Real-data examples may use plain arrays or data frames and are not required to construct this
  container.
- Dataset migrations must keep their reading and matrix-construction steps explicit rather than
  adding a generic runtime loader.
