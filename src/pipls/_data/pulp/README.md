# Pulp fiber-property regression data

These package resources support `pipls.datasets.load_pulp()`.

- `X.csv` contains 14 fiber-description predictors.
- `Y.csv` contains Canadian Standard Freeness and seven handsheet-property responses.
- `metadata.json` records labels, public provenance, preparation, dimensions, and integrity hashes.
- `LICENSE.txt` contains the attribution and CC BY 4.0 notice for this adapted dataset.

The public provenance reference is:

> Lindström, S. B., Ferritsius, R., Carlson, J. E., Persson, J., and Nilsson, F. (2025).
> Predicting handsheet properties and enhancing refiner control using fiber analyzer data and
> latent variable modeling. *Computers & Chemical Engineering*, **199**, 109143.
> [doi:10.1016/j.compchemeng.2025.109143](https://doi.org/10.1016/j.compchemeng.2025.109143).

The package adaptation selects the named fiber-property variables as predictors and the named pulp
and handsheet properties as responses. Refiner-control variables, specific refining energy, and the
final `k` variable are not part of the model matrices. All 46 rows, selected numeric values, column
order, and row order are preserved. No imputation, centering, scaling, row filtering, or learned
preprocessing is applied by the loader.
