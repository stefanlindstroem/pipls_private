# Datasets

Repository real datasets are added one at a time after source, citation, licensing, redistribution,
row-order, missing-value, public-provenance, and preparation choices are reviewed. Every dataset
directory uses
comma-delimited `X.csv`, comma-delimited `Y.csv`, and a documentary `metadata.yaml`.

The installed package provides the optional Phase E1 in-memory dataset container and deterministic
synthetic generator under `pipls.datasets`; see `docs/datasets.md`. Synthetic data are generated
at runtime and are not committed as dataset files.

Real-data examples read `X.csv` and `Y.csv` explicitly and form `X` and `Y` without a generic
package loader. `metadata.yaml` standardizes public repository descriptions and provenance, but it
is not read by the estimator and is not required for external users fitting their own data.
Private development paths and preparation-only scripts are not part of dataset integrations.

## Included datasets

- `pulp/`: CC BY 4.0 thermomechanical-pulp fiber-property and handsheet dataset.
- `sugarcane/`: CC BY 4.0 LabSpec absorbance spectra with four sugarcane quality responses.
- `tobacco/`: CC BY 4.0 raw FT-NIR spectra with 13 tobacco chemical responses.

Each example reads `X.csv` and `Y.csv` directly. `metadata.yaml` documents the repository asset but is not a runtime model input.
