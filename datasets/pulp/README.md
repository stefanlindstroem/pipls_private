# Pulp fiber-property regression data

This directory contains the analysis-facing matrices used for the Pi-PLS pulp example.

- `X.csv` contains 14 fiber-description predictors.
- `Y.csv` contains Canadian Standard Freeness and seven handsheet-property responses.
- `metadata.yaml` records the source, column selection, row alignment, license, and integrity hashes.
- `LICENSE.txt` contains the attribution and CC BY 4.0 notice for this adapted dataset.

The matrices were extracted from `data/pulp.csv` in the supplied `PiPLSR_v0.1` research archive.
The source table contains additional refiner-control variables, specific refining energy, and one
unused optical coefficient. The repository preparation keeps the exact 46-row order and selects
the same predictor and response columns as the archived Pi-PLS analysis code.

No imputation, centering, scaling, row filtering, or learned preprocessing was applied. The example
`examples/10_pulp_real_data.py` reads `X.csv` and `Y.csv` directly; it does not parse the metadata
file or use a package-owned loader.
