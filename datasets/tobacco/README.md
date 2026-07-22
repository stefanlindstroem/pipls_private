# Tobacco leaf FT-NIR dataset

This directory contains analysis-facing matrices for multi-output regression of 13 chemical
components from raw Fourier-transform near-infrared spectra of 347 tobacco leaf samples.

## Files

- `X.csv`: 347 rows and 1,557 raw spectral predictors. Column headers are wavenumbers in `cm^-1`,
  ordered from approximately 10,001 down to 4,000.
- `Y.csv`: the same 347 rows and 13 chemical response columns.
- `metadata.yaml`: public provenance, alignment, variable, preparation, license, and integrity
  documentation. It is not needed to fit Pi-PLS.
- `LICENSE.txt`: Creative Commons Attribution 4.0 license text.

## Public source

The original source is Chen, Guo, Wang, and Zhao (2025), *A Near-Infrared Spectroscopy
Dataset for Chemical Composition Prediction and Origin Identification of Tobacco Leaves*,
*Mendeley Data*, Version 1,
[doi:10.17632/9z7dgdtggk.1](https://doi.org/10.17632/9z7dgdtggk.1).

The related article is Chen et al. (2026), *A dataset for geographical origin identification of
tobacco leaves from multiple countries using near-infrared spectroscopy and chemometric
analysis*, *Data in Brief*, **64**, 112418,
[doi:10.1016/j.dib.2025.112418](https://doi.org/10.1016/j.dib.2025.112418).

The public collection provides one workbook containing spectra and sample metadata and one
workbook containing the 13 quantitative chemical reference values. The source collection is
licensed CC BY 4.0.

## Repository adaptation

The two public tables were matched one-to-one by their shared sample ID and ordered by that ID.
All 347 samples matched and neither table contains missing model values. The committed predictor
matrix excludes the sample ID, cultivation year, geographical origin, and duplicate spectrum-index
metadata columns. The response matrix retains all 13 chemical-component columns.

No smoothing, derivatives, multiplicative scatter correction, standard-normal-variate correction,
centering, scaling, imputation, or other spectral preprocessing was applied. The numeric values and
source column order are retained.

The executable example `examples/12_tobacco_real_data.py` reads `X.csv` and `Y.csv` directly with
pandas, evaluates a full-SVD Pi-PLS path, fits one selected fixed model, calculates OOF and
inspection results in memory, and writes final PDF figures with source-order response pagination.
It does not use a package loader or parse `metadata.yaml`.
