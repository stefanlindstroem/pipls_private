# Transparent real-data input contract

## Purpose

Pi-PLS is a regression package, not a general data-access framework. A programming user must be
able to read predictors `X`, read responses `Y`, and fit the model without a registry, metadata
parser, package-owned loader, or access to repository-development materials.

The package must not require a registry, metadata file, checksum manifest, dataset container, or
package-owned loader before a model can be fitted. A named built-in example dataset may be offered
as an optional convenience when an explicit decision assigns package ownership; it does not change
the primary user-owned `X`/`Y` contract.

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
synthetic generators and named reference datasets. It is not required for real data, and examples
must not imply otherwise. Decision 0142 assigns package ownership to Pulp, Sugarcane, and Tobacco
through three named loaders, with their resources included in the base installation. This creates
neither a general data-access extra nor a registry. During the accepted transition, only Pulp is
implemented as a package loader; Sugarcane and Tobacco continue to use repository CSV files until
their resource and consumer patches land.

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

Do not hide user-owned data preparation behind a generic utility such as `load_dataset`, an example
helper module, a registry resolver, metadata parser, or an implicit converter. A decision-authorized
named built-in dataset loader may replace repository file reading for that dataset only.

For repository datasets, the preferred visible pattern is:

```python
import pandas as pd

X = pd.read_csv("datasets/example/X.csv")
Y = pd.read_csv("datasets/example/Y.csv")
model = PiPLSRegression(n_components=2, predictor_rank=2).fit(X, Y)
```

The example must not need to parse `metadata.yaml`; that file documents the repository asset.

## Package-owned reference datasets

Decision 0138 implemented the first named package-owned dataset. Decision 0142 extends the final
closed set to Pulp, Sugarcane, and Tobacco. The currently implemented loader exposes:

```python
from pipls.datasets import load_pulp

data = load_pulp()
X, Y = load_pulp(return_X_y=True)
```

Each named loader returns the existing immutable `PiPLSDataset` or fresh read-only arrays from
installed package resources. Loaders are optional, local, and dataset-specific: no registry,
download, `as_frame` mode, pandas/PyYAML runtime dependency, or required loader protocol follows
from them. General users and every other real dataset continue to supply `X` and `Y` directly.

`load_pulp()` and its sole active package resources are implemented, and every maintained Pulp
consumer uses them. Its public wrapper now delegates to dataset-neutral private resource, metadata,
CSV, integrity, provenance, and sample-identifier machinery. `load_sugarcane()` and
`load_tobacco()` are accepted targets but are not added until Decision 0142 Patches 3 and 4. Their
current repository matrices remain temporary parity sources until consumer migration and final
duplicate removal.

The final resource directories under `src/pipls/_data/<dataset>/` are intentionally ordinary
CSV, JSON, README, and license assets. Public documentation must identify their locations in a
tagged source release, source distribution, wheel, and installed package so programming users in
R, C++, MATLAB, Julia, or another environment can use the exact loader matrices without Python.

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

The repository-level BSD 3-Clause License covers repository-authored code and documentation; it
does not relicense included datasets. Dataset-specific license and attribution files remain
authoritative for those assets.

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
- **Package-owned reference integration:** provides one canonical language-neutral resource set,
  named Python loading, public documentary metadata, provenance, and licenses.
- **Examples:** use the named loader for a package-owned reference dataset and show all later
  analytical choices; examples for user-owned data continue to form `X` and `Y` visibly.

## Prohibited directions

Do not introduce merely for repository examples:

- a public dataset registry;
- a generic real-data loader; the accepted named `load_pulp()`, `load_sugarcane()`, and
  `load_tobacco()` set does not authorize one;
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
ordinary-PLS paths; Pulp, Sugarcane, and Tobacco expose the conditional predictor-rank profile at
their selected component count. Tobacco applies search-owned one-standard-error selection, uses
the returned count for the rank profile, and retains deterministic source-order response
pagination through multipage PDFs.
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

The direct Pulp example and tutorial renderer both use the public `load_pulp()` API. The renderer
records the packaged dataset identifier, version, source DOI, license, resource hashes, canonical
array hashes, selected rank pair, evaluated predictor ranks, and boundary status in a generated
manifest and derives SVG figures from in-memory results. All tutorial figures and manifests are
ignored build products, not alternative dataset representations or package inputs.
