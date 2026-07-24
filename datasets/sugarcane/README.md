# Sugarcane LabSpec regression data

This directory contains analysis-facing matrices derived from the following public sugarcane
spectroscopy collection:

> Chaix, G., Bendoula, R., and Zgouz, A. (2021). Data set of Visible-Near Infrared handled and
> micro-spectrometers -- comparison of their accuracy for predicting some sugarcane properties.
> *Mendeley Data*, Version 1.
> [doi:10.17632/mjttsjfj2s.1](https://doi.org/10.17632/mjttsjfj2s.1).

The related data paper is Zgouz et al. (2020),
[doi:10.1016/j.dib.2020.106013](https://doi.org/10.1016/j.dib.2020.106013).

- `X.csv` contains LabSpec absorbance values at integer wavelengths from 780 through 2500 nm.
- `Y.csv` contains total sugar (`TS`), crude protein (`CP`), acid detergent fiber (`ADF`), and in
  vitro organic matter digestibility (`IVOMD`).
- `metadata.yaml` records the public source, exact row and wavelength selection, alignment,
  license, and integrity hashes.
- `LICENSE.txt` records attribution and the CC BY 4.0 data license.

The public source provides `LabSpec.csv` and `responses.csv`, both keyed by `Sample`. The repository
adaptation parses the four response columns, excludes samples 103, 105, and 111 because their `TS`
values are missing, joins the remaining rows one-to-one by `Sample`, and retains LabSpec wavelengths
from 780 through 2500 nm. The resulting matrices contain 57 aligned rows, 1,721 predictors, and four
responses.

No response values are imputed. No smoothing, derivative, scatter correction, centering, scaling,
or other spectral preprocessing is applied. `examples/06_sugarcane_real_data.py` reads `X.csv` and
`Y.csv` directly and runs ordinary Pi-PLS path selection; it does not parse the metadata or call a
package data loader.
