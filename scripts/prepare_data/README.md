# Dataset preparation scripts

Add deterministic source-to-analysis converters one dataset at a time when conversion is needed.
These scripts may verify sources, licenses, checksums, row order, and preparation choices for
repository reproducibility.

Preparation scripts are not runtime loaders. Examples and paper-reproduction scripts must read the
resulting analysis-facing files explicitly, form `X` and `Y` visibly, and then call the public
estimator API.

- `prepare_pulp.py`: verifies the supplied `PiPLSR_v0.1/data/pulp.csv` source and writes the
  documented 14-column `X.csv` and eight-column `Y.csv` analysis tables.
