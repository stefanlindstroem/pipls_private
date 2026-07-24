# Dataset API and generators

`pipls.datasets` contains optional immutable containers and deterministic synthetic-data
generators. They are conveniences for examples, tests, and structured experiments; ordinary arrays
and data frames passed directly to `fit(X, Y)` remain the normal real-data interface. Synthetic
truth stores only loading blocks that contribute to the generated predictor or response signal.

The repository also ships the Pulp, Sugarcane, and Tobacco [reference datasets](../datasets.md) and
[complete real-data examples](../examples.md#complete-real-data-analyses). Those datasets are CSV
assets read explicitly by the examples; they are not registry entries and are not loaded through
`PiPLSDataset`.

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

`PiPLSSyntheticTruth` is returned as `dataset.truth` by the generators; users normally inspect
its fields rather than construct it directly.

::: pipls.datasets.PiPLSSyntheticTruth
    options:
      show_signature: false
      members:
        - n_shared
        - n_predictor_specific
        - n_response_specific

## Generators

::: pipls.datasets.make_pipls_regression
    options:
      members: false

::: pipls.datasets.make_pipls_train_test
    options:
      members: false
