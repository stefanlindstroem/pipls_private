# Dataset interface and synthetic generator

Phase E1 introduces an optional structured in-memory dataset boundary for package-owned
synthetic data and experiments. Real-data users may pass ordinary arrays or data frames directly
to `fit(X, Y)`; no container or metadata file is required.

## Validated dataset container

```python
from pipls.datasets import PiPLSDataset

sample = PiPLSDataset(
    X=X,
    Y=Y,
    feature_names=("x_1", "x_2"),
    target_names=("y_1",),
    sample_ids=("sample_1", "sample_2"),
    provenance={
        "source": "local-example",
        "license": "BSD-3-Clause",
        "citation": "Example data",
        "version": "1",
    },
    metadata={"instrument": "example"},
)
```

The container validates and then freezes its contents:

- `X` is a finite two-dimensional real numeric array;
- `Y` is a finite one- or two-dimensional real numeric array and is stored as two-dimensional;
- rows of `X` and `Y` must align exactly;
- feature names, target names, and sample identifiers are non-empty, unique strings with matching
  lengths;
- provenance must contain non-empty `source`, `license`, `citation`, and `version` strings;
- metadata values may be immutable scalars, sequences, mappings, or NumPy arrays;
- arrays are copied, converted to `float64`, and made read-only;
- nested metadata mappings and sequences are frozen recursively.

The scikit-learn-style aliases `data` and `target` refer to `X` and `Y`. The container also exposes
`n_samples`, `n_features`, and `n_targets`.

## Deterministic latent-structure generator

```python
from pipls.datasets import make_pipls_regression

synthetic = make_pipls_regression(
    n_samples=200,
    n_features=40,
    n_targets=8,
    n_shared=3,
    n_predictor_specific=2,
    n_response_specific=1,
    shared_strength=(3.0, 2.0, 1.0),
    predictor_specific_strength=(1.5, 0.8),
    response_specific_strength=0.7,
    shared_distribution="normal",
    predictor_specific_distribution="uniform",
    response_specific_distribution="normal",
    feature_scale=1.0,
    target_scale=1.0,
    noise=(0.1, 0.2),
    random_state=0,
)
```

The generator uses a local `numpy.random.Generator`; it never changes NumPy's global random state.
A scalar `noise` applies to both blocks, while `(x_noise, y_noise)` controls them separately.
Strength and scale arguments accept either a scalar or one value per relevant latent direction or
observed variable.
Because latent score columns are centered, each generated sample block must contain more rows than
the larger of the declared predictor and response latent ranks. Invalid degenerate dimensions are
rejected rather than silently producing a lower-rank realization.

The generative model is

\[
X = T_s A_s^\mathsf{T} + T_x A_x^\mathsf{T} + E_x,
\qquad
Y = T_s B_s^\mathsf{T} + T_y B_y^\mathsf{T} + E_y,
\]

where `T_s` is shared, `T_x` is predictor-specific, and `T_y` is response-specific. The loading
columns are orthonormal within each observed block. Latent score columns are centered and scaled
to unit sample standard deviation after being drawn from the selected normal or uniform source
distribution.

`synthetic.truth` is a read-only `PiPLSSyntheticTruth` containing latent scores, loading blocks,
signal matrices, noise matrices, strengths, and observed-variable scales. Structurally impossible
loading blocks are present as explicit zeros, which makes invariants directly testable.

## Leakage-free train/test generation

```python
from pipls.datasets import make_pipls_train_test

train, test = make_pipls_train_test(
    n_train=150,
    n_test=50,
    n_features=40,
    n_targets=8,
    n_shared=3,
    n_predictor_specific=2,
    n_response_specific=1,
    random_state=0,
)
```

Both blocks share one generated set of loadings, strengths, and observed-variable scales. Their
latent scores and noise are independent draws. No centering, standardization, imputation, feature
selection, or other fitted preprocessing is applied across the train/test boundary. The training
block is unchanged when only `n_test` changes.

## Real-data boundary

Real-data reading remains user-owned. Examples and reproduction scripts must show how `X` and `Y`
are read and formed directly using ordinary NumPy, pandas, or domain-specific code. The project may
track preparation scripts, provenance, licenses, and checksums for its own datasets, but no public
registry, generic loader, or required metadata sidecar is part of the runtime API.
