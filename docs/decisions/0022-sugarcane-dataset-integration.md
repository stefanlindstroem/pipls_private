# Decision 0022: sugarcane dataset integration

## Status

Accepted during Phase E3.

## Context

The public sugarcane spectroscopy collection provides multiple instrument tables and a response
table for 60 samples. The Pi-PLS analysis uses the LabSpec spectrum, four continuous responses,
and a wavelength restriction. The integration must preserve the transparent `X`/`Y` programming
workflow while documenting the scientifically relevant row and variable selection.

A spectral matrix also demonstrates that enumerating thousands of nearly identical variable
descriptions in `metadata.yaml` is less informative than an exact compact axis description.

## Decision

- Integrate the public Mendeley Data collection identified by DOI `10.17632/mjttsjfj2s.1` as the
  third repository real dataset.
- Use the LabSpec absorbance table for predictors and `TS`, `CP`, `ADF`, and `IVOMD` for responses.
- Align source rows one-to-one by `Sample` while preserving LabSpec order.
- Exclude samples 103, 105, and 111 because their `TS` values are missing; do not impute them.
- Retain wavelengths from 780 through 2500 nm inclusive.
- Apply no smoothing, derivative, scatter correction, centering, scaling, or other spectral
  preprocessing when producing the committed matrices.
- Publish only the analysis-facing `X.csv`, `Y.csv`, public metadata, attribution notice, and a
  direct-reading example; do not add a package loader or preparation-only script.
- Permit `metadata.yaml` to describe a large regular predictor grid with an exact compact axis
  descriptor instead of repeating one entry for every column.

## Consequences

- The repository gains a moderate-size, high-dimensional spectral regression example under CC BY
  4.0 without broadening the runtime API.
- Users can see exactly which samples and wavelengths define the model matrices.
- The example remains `read X`, `read Y`, and fit the estimator.
- The metadata contract remains consistent at the top level while supporting scientifically clear
  descriptions of regular spectral axes.
- This integration is an example dataset, not a frozen benchmark or manuscript-result claim.
