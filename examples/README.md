# Examples

These concise executable examples demonstrate ordinary package use. They are not manuscript-figure
or publication-result workflows.

- `07_advanced_cv.py`: grouped and advanced cross-validation workflows.
- `08_synthetic_data.py`: deterministic train/test generation with shared latent structure.
- `09_linnerud_real_data.py`: explicit pandas reading of the Linnerud `X.csv` and `Y.csv`.
- `10_pulp_real_data.py`: explicit pandas reading and path selection for pulp `X.csv` and `Y.csv`.
- `11_sugarcane_real_data.py`: explicit pandas reading of high-dimensional LabSpec sugarcane
  spectra and four responses.
- `12_tobacco_real_data.py`: explicit pandas reading of raw FT-NIR tobacco spectra and 13 chemical
  responses.

Real-data examples must show the ordinary I/O used to create `X` and `Y` in the example itself.
Do not route example data through a package registry, generic loader, or hidden helper module.
