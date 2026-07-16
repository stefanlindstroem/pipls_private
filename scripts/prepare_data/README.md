# Dataset preparation scripts

Add deterministic source-to-analysis converters one dataset at a time when conversion is needed.
These scripts may verify sources, licenses, checksums, row order, and preparation choices for
repository reproducibility.

Preparation scripts are not runtime loaders. Examples and paper-reproduction scripts must read the
resulting analysis-facing files explicitly, form `X` and `Y` visibly, and then call the public
estimator API.
