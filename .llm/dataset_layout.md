# Repository real-dataset layout

## Purpose

This contract standardizes committed analysis-facing real datasets without changing the public
estimator API. External programming users still read and prepare their own `X` and `Y`; they do not
need a metadata file or package loader.

## Required files

Every committed real dataset directory under `datasets/<dataset-id>/` must contain:

- `X.csv`: predictor matrix;
- `Y.csv`: response matrix;
- `metadata.yaml`: repository description and provenance.

Additional human-readable files such as `README.md` and license files may be included when needed.
Preparation scripts belong under `scripts/prepare_data/`, not in runtime package code.

## CSV contract

`X.csv` and `Y.csv` must:

- be UTF-8 text;
- use a comma as delimiter;
- contain one header row;
- contain only analysis-facing numeric model columns;
- preserve a documented common row order;
- use the exact filenames `X.csv` and `Y.csv`.

Examples and reproduction scripts must read these files directly with ordinary NumPy or pandas
code. They must not call a package loader or parse `metadata.yaml` to construct the model matrices.

## Metadata contract

`metadata.yaml` is required for repository-included datasets but is not a runtime input. It must
use `schema_version: 1` and contain these top-level fields:

- `dataset`: stable identifier, title, version, and summary;
- `files`: `predictors: X.csv` and `responses: Y.csv`;
- `format`: CSV type, comma delimiter, UTF-8 encoding, and header status;
- `dimensions`: sample, predictor, and response counts;
- `predictors`: ordered names and non-empty descriptions;
- `responses`: ordered names and non-empty descriptions;
- `sample_alignment`: row-alignment method and explanation;
- `missing_values`: declared policy for predictors and responses;
- `source`: provider, upstream files or references, and citation;
- `license`: identifier, local license file when applicable, and redistribution status;
- `preparation`: transparent account of transformations from source to analysis files;
- `integrity`: SHA-256 hashes for the analysis files and relevant license assets.

The metadata may contain additional dataset-specific fields, but the required fields and meanings
must remain stable across datasets.

## Separation from the programming-user contract

Repository consistency does not imply a public data-management framework. The ordinary user
workflow remains:

```python
X = ...
Y = ...
model = PiPLSRegression().fit(X, Y)
```

A user may ignore `metadata.yaml`, use different filenames, use another delimiter, read a database,
or obtain arrays from any domain-specific source. The repository convention exists for review,
reproducibility, and consistent examples only.
