# Dataset API and generators

`pipls.datasets` contains named packaged real-data loaders, optional immutable containers, and
deterministic synthetic-data generators. They are conveniences for examples, tests, and structured
experiments; ordinary arrays and data frames passed directly to `fit(X, Y)` remain the normal
real-data interface. Synthetic truth stores only loading blocks that contribute to the generated
predictor or response signal.

Pulp and Sugarcane are available through the named package-owned loaders below. Tobacco remains a
[repository reference dataset](../datasets.md) read explicitly by its
[maintained example](../examples.md#complete-real-data-analyses). Sugarcane's maintained examples
continue to use the temporary repository copy until the consumer-migration patch; no generic
registry or download layer is provided.

Mathematical notation on this page follows the package convention: complete matrices are bold,
descriptive role and block subscripts are upright, and variable indices remain italic. For example,
$\boldsymbol{\Lambda}_{\mathrm{p}}$ and $\mathbf{L}_{\mathrm{sp}}$ are complete matrices,
while $d_k$ retains the variable index $k$.

## Packaged datasets

::: pipls.datasets.load_pulp
    options:
      members: false

`load_pulp()` performs no network access or preprocessing. Its default `PiPLSDataset` result keeps
labels, sample identifiers, provenance, and recursively frozen metadata; `return_X_y=True` returns
the same read-only predictor and response arrays directly.

::: pipls.datasets.load_sugarcane
    options:
      members: false

`load_sugarcane()` returns the 57 by 1,721 LabSpec predictor matrix and four aligned responses from
installed package resources. The feature names are the wavelength labels `"780"` through `"2500"`;
loading performs no network access or spectral preprocessing. The temporary repository copy remains
only for parity and later consumer migration.

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
