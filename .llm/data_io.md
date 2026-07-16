# Transparent real-data input contract

## Purpose

This document defines how Pi-PLS examples, repository datasets, and paper-reproduction scripts
must obtain real predictor and response matrices. It protects a deliberately small user contract:
the programming user reads their own data into `X` and `Y`, then calls the estimator.

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
3. show any row alignment, column selection, dtype conversion, or missing-value policy in the
   example itself;
4. form `X` and `Y` visibly;
5. call `fit(X, Y)`.

Do not hide these steps behind a package utility such as `load_dataset`, an example helper module,
a registry resolver, or an implicit converter. A reader should be able to inspect one script and
see exactly how the model matrices were formed.

Ordinary, recognizable I/O is preferred. For example:

```python
import numpy as np

X = np.loadtxt("data/X.csv", delimiter=",", skiprows=1)
Y = np.loadtxt("data/Y.csv", delimiter=",", skiprows=1)
```

or, when column names matter:

```python
import pandas as pd

frame = pd.read_csv("data/measurements.csv")
X = frame.loc[:, predictor_columns]
Y = frame.loc[:, response_columns]
```

The exact code may vary by dataset. Consistency across repository examples is desirable, but it
must not be achieved by concealing the I/O contract.

## Repository dataset policy

Repository-owned datasets may include human-readable descriptions, citations, licenses,
preparation notes, source checksums, and deterministic preparation scripts where needed for
scientific reproducibility. These are repository and publication assets, not runtime inputs that
external users must reproduce.

For each migrated dataset:

- document the source, license, citation, redistribution decision, and preparation choices;
- keep preparation code under `scripts/prepare_data/` when raw-to-analysis conversion is needed;
- keep the analysis-facing files simple enough to read explicitly in the example or reproduction
  script;
- state shapes, predictor columns, response columns, row ordering, and missing-value handling;
- preserve deterministic preparation and checksums where they are scientifically useful;
- do not expose a generic registry or loader API merely to support repository examples.

A metadata file may be used internally when a particular preparation workflow benefits from it,
but it must be optional for model use and must not become a prerequisite for reading `X` and `Y`.

## Separation of responsibilities

The boundaries are:

- **Programming user:** reads and prepares `X` and `Y` according to their domain and data source.
- **Pi-PLS estimators:** validate model inputs, fit, transform, predict, score, and report results.
- **Repository preparation scripts:** reproducibly convert specific research sources into simple
  analysis-facing files.
- **Examples and reproduction scripts:** show the actual reading and matrix construction steps
  directly.
- **Synthetic API:** may return `PiPLSDataset` because the package itself generates all arrays and
  latent truth.

## Explicit exclusions

Do not add, unless the project owner reverses this decision:

- a required metadata sidecar for fitting;
- a public dataset registry;
- a generic package loader for arbitrary real datasets;
- automatic download or converter execution from estimator or example code;
- hidden row filtering, imputation, centering, scaling, or feature engineering;
- example-only helper functions that obscure how `X` and `Y` were read.

## Review questions

For any real-data or example patch, verify:

- Can a reader see exactly how `X` and `Y` are obtained?
- Could the same estimator code accept arrays loaded another way?
- Is all dataset-specific handling outside the estimator API?
- Are preparation and analysis-time reading clearly separated?
- Is optional provenance documentation being mistaken for required runtime metadata?
