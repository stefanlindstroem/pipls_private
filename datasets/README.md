# Datasets

Repository real datasets are added one at a time after source, citation, licensing, redistribution,
row-order, missing-value, and preparation choices are reviewed. Every dataset directory uses
comma-delimited `X.csv`, comma-delimited `Y.csv`, and a documentary `metadata.yaml`.

Available integrations:

- `linnerud/`: 20 exercise observations and three physiological responses, copied from
  scikit-learn under the BSD 3-Clause license. See `datasets/linnerud/README.md` and
  `examples/09_linnerud_real_data.py`.

The installed package provides the optional Phase E1 in-memory dataset container and deterministic
synthetic generator under `pipls.datasets`; see `docs/datasets.md`. Synthetic data are generated
at runtime and are not committed as dataset files.

Real-data examples read `X.csv` and `Y.csv` explicitly and form `X` and `Y` without a generic
package loader. `metadata.yaml` standardizes repository descriptions and provenance, but it is not
read by the estimator and is not required for external users fitting their own data.

## Included datasets

- `linnerud/`: small BSD-licensed physical-exercise reference dataset.
- `pulp/`: CC BY 4.0 thermomechanical-pulp fiber-property and handsheet dataset.

Each example reads `X.csv` and `Y.csv` directly. `metadata.yaml` documents the repository asset but is not a runtime model input.
