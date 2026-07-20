# Dataset containers and synthetic data

`pipls.datasets` contains optional immutable containers and deterministic synthetic-data
generators. They are conveniences for examples, tests, and structured experiments; ordinary arrays
and data frames passed directly to `fit(X, Y)` remain the normal real-data interface.

## Containers

::: pipls.datasets.PiPLSDataset
    options:
      members:
        - data
        - target
        - n_samples
        - n_features
        - n_targets

::: pipls.datasets.PiPLSSyntheticTruth
    options:
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
