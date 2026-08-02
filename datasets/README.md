# Datasets

Repository real datasets are added one at a time after source, citation, licensing, redistribution,
row-order, missing-value, public-provenance, and preparation choices are reviewed. Every active
repository dataset directory uses comma-delimited `X.csv`, comma-delimited `Y.csv`, and a
documentary `metadata.yaml`.

The installed package separately provides the named `pipls.datasets.load_pulp()` dataset, the
optional in-memory dataset container, and deterministic synthetic generators; see
`docs/datasets.md`. Synthetic data are generated at runtime and are not committed as dataset files.
No generic loader or registry is provided.

Repository real-data examples read `X.csv` and `Y.csv` explicitly. `metadata.yaml` standardizes
public descriptions and provenance, but it is not read by the estimator and is not required for
external users fitting their own data. Private development paths and preparation-only scripts are
not part of dataset integrations.

## Included repository datasets

- `sugarcane/`: CC BY 4.0 LabSpec absorbance spectra with four sugarcane quality responses from
  [Chaix, Bendoula, and Zgouz (2021)](https://doi.org/10.17632/mjttsjfj2s.1); see also the
  [related data paper](https://doi.org/10.1016/j.dib.2020.106013).
- `tobacco/`: CC BY 4.0 raw FT-NIR spectra with 13 tobacco chemical responses from
  [Chen, Guo, Wang, and Zhao (2025)](https://doi.org/10.17632/9z7dgdtggk.1); see also the
  [related data paper](https://doi.org/10.1016/j.dib.2025.112418).

Each repository example reads `X.csv` and `Y.csv` directly. `metadata.yaml` documents the asset but
is not a runtime model input. Full source citations and preparation details are given in each
dataset README and metadata file.
