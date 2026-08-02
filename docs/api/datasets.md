# Dataset API and generators

`pipls.datasets` contains one packaged real-data loader, optional immutable containers, and
deterministic synthetic-data generators. They are conveniences for examples, tests, and structured
experiments; ordinary arrays and data frames passed directly to `fit(X, Y)` remain the normal
real-data interface. Synthetic truth stores only loading blocks that contribute to the generated
predictor or response signal.

Pulp is available through the named package-owned loader below. Sugarcane and Tobacco remain
[repository reference datasets](../datasets.md) read explicitly by their
[maintained examples](../examples.md#complete-real-data-analyses); no generic registry or download
layer is provided.

Mathematical notation on this page follows the package convention: complete matrices are bold,
descriptive role and block subscripts are upright, and variable indices remain italic. For example,
$\boldsymbol{\Lambda}_{\mathrm{p}}$ and $\mathbf{L}_{\mathrm{sp}}$ are complete matrices,
while $d_k$ retains the variable index $k$.

## Packaged dataset

::: pipls.datasets.load_pulp
    options:
      members: false

`load_pulp()` performs no network access or preprocessing. Its default `PiPLSDataset` result keeps
labels, sample identifiers, provenance, and recursively frozen metadata; `return_X_y=True` returns
the same read-only predictor and response arrays directly.

## Containers

::: pipls.datasets.PiPLSDataset
    options:
      members:
        - data
        - target
        - n_samples
        - n_features
        - n_targets

Metadata NumPy arrays are copied and made read-only, but they must have a non-object dtype. Use
ordinary nested sequences and mappings for heterogeneous metadata so every nested value can be
validated and frozen explicitly.

`PiPLSRegressionTruth` is returned by the configurable package generators, while
`PiPLSLatentGeometryTruth` is returned by the companion-manuscript generator. Users normally
inspect these records through `dataset.truth` rather than construct them directly.

::: pipls.datasets.PiPLSRegressionTruth
    options:
      show_signature: false
      members:
        - n_shared
        - n_predictor_specific
        - n_response_specific

::: pipls.datasets.PiPLSLatentGeometryTruth
    options:
      show_signature: false
      members:
        - n_shared
        - n_predictor_specific
        - n_response_specific

## Generators

The [companion-manuscript synthetic-data guide](../manuscript_reproduction.md) shows how to use the
exact Gaussian latent generator, inspect its truth matrices, and distinguish distribution-level
from realization-level and complete-study reproduction.

::: pipls.datasets.make_pipls_latent_geometry
    options:
      members: false

::: pipls.datasets.make_pipls_regression
    options:
      members: false

::: pipls.datasets.make_pipls_train_test
    options:
      members: false
