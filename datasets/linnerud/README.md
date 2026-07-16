# Linnerud physical-exercise dataset

This directory contains the first transparent real-data integration for the Pi-PLS repository.

## Data

The dataset contains observations from 20 middle-aged men in a fitness club.

- `exercise.csv`: three predictor columns — `Chins`, `Situps`, and `Jumps`.
- `physiological.csv`: three response columns — `Weight`, `Waist`, and `Pulse`.

Both files contain 20 rows in the same documented sample order. There are no missing values.
The tables are intentionally kept as separate, simple text files so the example can show the
ordinary programming-user workflow: read `X`, read `Y`, verify alignment, and fit.

## Source and citation

The files are copied verbatim from scikit-learn 1.8.0:

- `sklearn/datasets/data/linnerud_exercise.csv`
- `sklearn/datasets/data/linnerud_physiological.csv`

The scikit-learn dataset description cites:

Tenenhaus, M. (1998). *La régression PLS: théorie et pratique*. Paris: Editions Technip.

The original exercise study is commonly attributed to Linnerud. This repository does not infer
participant identifiers, units beyond the published column names, or additional preprocessing.

## License and redistribution

The copied files are redistributed under scikit-learn's BSD 3-Clause license. See `LICENSE.txt`.
The repository records file hashes in `checksums.sha256` for integrity; the example does not require
or read that file.

## Analysis-time contract

`examples/09_linnerud_real_data.py` reads both tables directly with pandas. It checks:

- equal row counts;
- expected predictor and response column names;
- numeric dtypes;
- absence of missing values.

It then forms `X` and `Y` visibly and calls `PiPLSRegression.fit(X, Y)`. No package loader,
registry, metadata sidecar, or hidden example helper is involved.
