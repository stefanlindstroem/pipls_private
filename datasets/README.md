# Datasets

Repository real datasets are added one at a time after source, citation, licensing, redistribution,
row-order, missing-value, and preparation choices are reviewed.

Available integrations:

- `linnerud/`: 20 exercise observations and three physiological responses, copied from
  scikit-learn under the BSD 3-Clause license. See `datasets/linnerud/README.md` and
  `examples/09_linnerud_real_data.py`.

The installed package provides the optional Phase E1 in-memory dataset container and deterministic
synthetic generator under `pipls.datasets`; see `docs/datasets.md`. Synthetic data are generated
at runtime and are not committed as dataset files.

Real-data examples read predictor and response files explicitly and form `X` and `Y` without a
generic package loader. Human-readable provenance, licenses, and integrity hashes may accompany a
repository dataset, but they are not required for external users fitting their own data.
