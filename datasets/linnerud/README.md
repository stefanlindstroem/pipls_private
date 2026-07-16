# Linnerud physical-exercise dataset

This directory contains the first transparent real-data integration for the Pi-PLS repository.

## Repository dataset layout

All committed real datasets use the same analysis-facing names and format:

- `X.csv`: predictor matrix, comma-delimited with a header row;
- `Y.csv`: response matrix, comma-delimited with a header row;
- `metadata.yaml`: repository description, source, license, dimensions, variables, alignment,
  preparation, and integrity hashes.

`metadata.yaml` is a repository asset. It is not read by the estimator or required by external
users. The executable example deliberately reads only `X.csv` and `Y.csv`.

## Data

The dataset contains observations from 20 middle-aged men in a fitness club.

- `X.csv`: `Chins`, `Situps`, and `Jumps`.
- `Y.csv`: `Weight`, `Waist`, and `Pulse`.

Both files contain 20 rows in the same documented positional order. There are no missing values.

## Source and citation

The values are copied from scikit-learn 1.8.0:

- `sklearn/datasets/data/linnerud_exercise.csv`
- `sklearn/datasets/data/linnerud_physiological.csv`

The source dataset description cites:

Tenenhaus, M. (1998). *La regression PLS: theorie et pratique*. Paris: Editions Technip.

## License and preparation

The data are redistributed under scikit-learn's BSD 3-Clause license; see `LICENSE.txt`.
The upstream whitespace-delimited tables were converted to comma-delimited CSV and renamed to
`X.csv` and `Y.csv`. Values, columns, and row order were not changed. The preparation and SHA-256
hashes are recorded in `metadata.yaml`.

## Analysis-time contract

`examples/09_linnerud_real_data.py` reads `X.csv` and `Y.csv` directly with pandas, checks columns,
row alignment, numeric dtypes, and missingness, then calls `PiPLSRegression.fit(X, Y)`. No package
loader, registry, metadata parser, or hidden helper is involved.
