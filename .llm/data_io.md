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

model = PiPLSRegression(n_components=2, predictor_rank=2).fit(X, Y)
```

`X` and `Y` may come from NumPy, pandas, a database client, a domain-specific file reader, or any
other user-controlled source compatible with the public estimator validation rules. Pi-PLS owns
the regression behavior after `X` and `Y` are supplied; it does not own general-purpose data
access.

`PiPLSDataset` remains an optional structured container and the return type of the package-owned
synthetic generators. It is not required for real data, and examples must not imply otherwise.

## Example transparency

Examples must behave as an ordinary programming user is expected to behave:

1. read the predictor file or columns explicitly;
2. read the response file or columns explicitly;
3. show any scientifically meaningful row alignment or column selection in the script;
4. form `X` and `Y` visibly;
5. call `fit(X, Y)`.

For committed repository datasets, examples may trust the documented CSV schema and tested file
layout. Do not repeat dtype, missing-value, header-order, or directory-existence checks when the
repository already establishes those facts. External users remain responsible for validating their
own data sources.

Do not hide these steps behind a package utility such as `load_dataset`, an example helper module,
a registry resolver, metadata parser, or an implicit converter.

For repository datasets, the preferred visible pattern is:

```python
import pandas as pd

X = pd.read_csv("datasets/example/X.csv")
Y = pd.read_csv("datasets/example/Y.csv")
model = PiPLSRegression(n_components=2, predictor_rank=2).fit(X, Y)
```

The example must not need to parse `metadata.yaml`; that file documents the repository asset.

## Public provenance boundary

Everything committed with a repository dataset is potentially visible to programming users.
Dataset metadata, README files, examples, licenses, and decision records therefore use only:

- publicly accessible publications, repositories, archives, or included raw files;
- public citations, DOI values, resolvable DOI URLs, and license statements;
- analysis-facing transformations that a user can understand from the committed materials.

Do not publish personal delivery details, private archive names, inaccessible local paths,
checksums of unshared source files, or preparation scripts that only reconstruct committed files
from private development inputs. Such development-only lineage remains outside the public
repository.

## Licensing gate

A dataset is eligible for repository inclusion only when the exact source material used to create
the committed tables has an explicit license or written permission that permits redistribution and
adaptation for general use. Public accessibility, an academic citation, or a license attached only
to a different derivative does not satisfy this requirement.

The source license and required attribution must be preserved in the dataset directory and recorded
in `metadata.yaml`. If rights are uncertain, the data remain external to `pipls`; an example may not
work around this boundary by downloading and republishing the same material automatically.

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
The served dataset guide reproduces each original-source citation and DOI link so provenance is
available without opening the documentary YAML file.

## Public reconstruction exception

A reconstruction or preprocessing script is appropriate only when all of the following hold:

- the raw source is itself included or publicly obtainable;
- the transformation is scientifically relevant to how users should form `X` and `Y`;
- the code is intended to be read and run by programming users;
- the example exposes, rather than hides, the analytical choices.

No current dataset uses this exception. A future reconstruction must first pass the repository
licensing gate and must remain dataset-specific rather than becoming a generic internal preparation
layer.

## Separation of responsibilities

- **Programming user:** reads and prepares `X` and `Y` from their own source.
- **Estimator API:** validates supplied model matrices and fits Pi-PLS.
- **Repository dataset integration:** provides consistently named analysis files plus public
  documentary metadata and provenance.
- **Examples:** visibly read `X.csv` and `Y.csv` and show all analytical choices that form the
  matrices.

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

## Analysis artifact transparency

Committed `X.csv` and `Y.csv` files are input assets, not a reason to serialize intermediate
analysis results. Every numbered real-data workflow keeps its path and inspection results in memory,
creates figures directly, and writes only final PDF outputs. Example 04 compares immutable Pi-PLS and
ordinary-PLS paths; Pulp additionally exposes the conditional predictor-rank profile for its chosen
component count; Tobacco owns deterministic source-order response pagination through multipage PDFs.
New numbered-example work should follow that pattern. Git and source distributions preserve the
required output-directory structure through `.gitkeep` files; generated PDFs are never committed
or included in snapshots. The synthetic leave-one-out example writes no artifact: it reports one
compact validation result directly and leaves application-specific reporting to the user.

## Publication boundary

Paper-specific data orchestration belongs in downstream reproduction repositories that depend on a
tagged `pipls` release. This repository's examples remain concise package-use examples and should
not grow into manuscript pipelines, figure generation, or complete comparison grids.

## Tutorial assets

The synthetic tutorial uses `make_pipls_train_test()` directly in example 02 and its renderer. Its
manifest records the deterministic generator configuration, selected rank pair, external-test
provenance, and SVG hashes. No generated table is an input to fitting or plotting.

The direct Pulp example and tutorial renderer both read the committed `datasets/pulp/X.csv` and
`Y.csv` tables directly. The renderer records their SHA-256 values, selected rank pair, evaluated
predictor ranks, and boundary status in a generated manifest and derives SVG figures from in-memory
results. All tutorial figures and manifests are ignored build products, not alternative dataset
representations or package inputs.
