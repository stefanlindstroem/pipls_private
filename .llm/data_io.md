# Transparent real-data input contract

## Purpose

This document defines how Pi-PLS examples, repository datasets, and paper-reproduction scripts
obtain real predictor and response matrices. It protects a deliberately small user contract: the
programming user reads their own data into `X` and `Y`, then calls the estimator.

The package must not require a registry, metadata file, checksum manifest, dataset container, or
package-owned loader before a model can be fitted.

## Programming-user contract

The primary public workflow is:

```python
X = ...  # user-owned data reading and selection
Y = ...

model = PiPLSRegression().fit(X, Y)
```

`X` and `Y` may come from NumPy, pandas, a database client, a domain-specific file reader, or any
other user-controlled source compatible with the public estimator validation rules. Pi-PLS owns
the regression behavior after `X` and `Y` are supplied; it does not own general-purpose data
access.

`PiPLSDataset` remains an optional structured container and the return type of the package-owned
synthetic generators. It is not required for real data, and examples must not imply otherwise.

## Example transparency

Examples and paper-reproduction scripts must behave as an ordinary programming user is expected
to behave:

1. read the predictor file or columns explicitly;
2. read the response file or columns explicitly;
3. show row alignment, column selection, dtype conversion, and missing-value policy in the script;
4. form `X` and `Y` visibly;
5. call `fit(X, Y)`.

Do not hide these steps behind a package utility such as `load_dataset`, an example helper module,
a registry resolver, metadata parser, or an implicit converter.

For repository datasets, the preferred visible pattern is:

```python
import pandas as pd

X = pd.read_csv("datasets/example/X.csv")
Y = pd.read_csv("datasets/example/Y.csv")
model = PiPLSRegression().fit(X, Y)
```

The example must not need to parse `metadata.yaml`; that file documents the repository asset.

## Repository dataset policy

Every committed real dataset follows `.llm/dataset_layout.md`:

- predictors are stored in UTF-8, comma-delimited `X.csv` with a header;
- responses are stored in UTF-8, comma-delimited `Y.csv` with a header;
- `metadata.yaml` records a consistent description, source, license, dimensions, variables,
  alignment, preparation, missing-value policy, and integrity hashes;
- the metadata file is mandatory for repository inclusion but optional and irrelevant for model
  fitting by external users;
- human-readable descriptions and license files may accompany the standard files;
- deterministic source-to-analysis conversion belongs under `scripts/prepare_data/` when needed.

Repository metadata standardizes scientific assets; it does not create a public registry or loader.

## Separation of responsibilities

- **Programming user:** reads and prepares `X` and `Y` from their own source.
- **Estimator API:** validates supplied model matrices and fits Pi-PLS.
- **Repository dataset integration:** provides consistently named analysis files plus documentary
  metadata and provenance.
- **Examples and reproduction scripts:** visibly read `X.csv` and `Y.csv` and show all analytical
  choices that form the matrices.

## Prohibited directions

Do not introduce merely for repository examples:

- a public dataset registry;
- a generic real-data loader;
- automatic downloading;
- runtime dependence on `metadata.yaml`;
- hidden example helpers that conceal how `X` and `Y` were formed;
- preprocessing learned across train/test or cross-validation boundaries.
