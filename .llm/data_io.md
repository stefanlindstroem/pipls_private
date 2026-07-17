# Transparent real-data input contract

## Purpose

Pi-PLS is a regression package, not a general data-access framework. A programming user must be
able to read predictors `X`, read responses `Y`, and fit the model without a registry, metadata
parser, package-owned loader, or access to repository-development materials.

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

## Public provenance boundary

Everything committed with a repository dataset is potentially visible to programming users.
Dataset metadata, README files, examples, licenses, and decision records therefore use only:

- publicly accessible publications, repositories, archives, or included raw files;
- public citations, DOI values, URLs, and license statements;
- analysis-facing transformations that a user can understand from the committed materials.

Do not publish personal delivery details, private archive names, inaccessible local paths,
checksums of unshared source files, or preparation scripts that only reconstruct committed files
from private development inputs. Such development-only lineage remains outside the public
repository.

## Repository dataset policy

Every committed real dataset follows `.llm/dataset_layout.md`:

- predictors are stored in UTF-8, comma-delimited `X.csv` with a header;
- responses are stored in UTF-8, comma-delimited `Y.csv` with a header;
- `metadata.yaml` records a consistent public description, source, license, dimensions, variables,
  alignment, preparation, missing-value policy, and integrity hashes;
- the metadata file is mandatory for repository inclusion but optional and irrelevant for model
  fitting by external users;
- human-readable descriptions and license files may accompany the standard files.

Repository metadata standardizes scientific assets; it does not create a public registry or loader.

## Public reconstruction exception

A reconstruction or preprocessing script is appropriate only when all of the following hold:

- the raw source is itself included or publicly obtainable;
- the transformation is scientifically relevant to how users should form `X` and `Y`;
- the code is intended to be read and run by programming users;
- the example exposes, rather than hides, the analytical choices.

Corn is the planned special case: its raw public data and analysis-relevant reading and
preprocessing must be shown explicitly. That code belongs with the public example or dataset
materials, not in a generic internal preparation layer.

## Separation of responsibilities

- **Programming user:** reads and prepares `X` and `Y` from their own source.
- **Estimator API:** validates supplied model matrices and fits Pi-PLS.
- **Repository dataset integration:** provides consistently named analysis files plus public
  documentary metadata and provenance.
- **Examples and reproduction scripts:** visibly read `X.csv` and `Y.csv` and show all analytical
  choices that form the matrices.

## Prohibited directions

Do not introduce merely for repository examples:

- a public dataset registry;
- a generic real-data loader;
- automatic downloading;
- runtime dependence on `metadata.yaml`;
- hidden example helpers that conceal how `X` and `Y` were formed;
- private archive or local-path references in committed dataset materials;
- preparation-only scripts whose required source is unavailable to users;
- preprocessing learned across train/test or cross-validation boundaries.
