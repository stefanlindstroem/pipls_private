# Transparent real-data input contract

## Purpose

Pi-PLS is a regression package, not a general data-access framework. A programming user must be
able to read predictors `X`, read responses `Y`, and fit the model without a registry, metadata
parser, package-owned loader, or access to repository-development materials.

Named reference-dataset loaders are optional conveniences for three decision-owned examples. They
do not change the primary user-owned `X`/`Y` contract.

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
access. `PiPLSDataset` is optional and must never be presented as a prerequisite for fitting.

## Example transparency

Examples must behave as an ordinary programming user is expected to behave:

1. obtain predictors and responses through a visible user-owned read or an explicitly named
   package-owned loader;
2. show scientifically meaningful row alignment or column selection when it is part of the user
   workflow rather than already fixed in a package-owned resource;
3. form `X` and `Y` visibly;
4. call `fit(X, Y)`.

Do not hide user-owned data preparation behind a generic utility such as `load_dataset`, a registry
resolver, metadata parser, or implicit converter. A decision-authorized named loader may replace
file reading for its dataset only.

## Package-owned reference datasets

Decision 0142 establish the final closed set of named reference datasets:

```python
from pipls.datasets import load_pulp, load_sugarcane, load_tobacco

pulp = load_pulp()
sugarcane = load_sugarcane()
tobacco = load_tobacco()
X, Y = load_tobacco(return_X_y=True)
```

Each loader returns the immutable `PiPLSDataset` or fresh read-only `float64` arrays from installed
package resources. Loading performs no download, imputation, centering, scaling, row filtering,
spectral preprocessing, or model fitting. There is no registry, `as_frame` mode, optional data
extra, pandas/PyYAML runtime dependency, or generic loader protocol.

The sole active resources are under `src/pipls/_data/<dataset>/` and contain `X.csv`, `Y.csv`,
`metadata.json`, `README.md`, and `LICENSE.txt`. The CSV pairs are exactly the matrices returned by
their loaders. Public documentation identifies the corresponding locations in tagged source
releases, source distributions, wheels, and installed packages so R, C++, MATLAB, Julia, and other
users can consume the same files without Python.

## Public provenance and licensing boundary

Everything committed with a reference dataset is public-facing. Metadata, README files, examples,
licenses, and decision records therefore use only:

- publicly accessible publications, repositories, archives, or included raw files;
- public citations, DOI values, resolvable DOI URLs, and license statements;
- analysis-facing transformations that a user can understand from the committed materials.

A dataset is eligible for package inclusion only when the exact source material used to create the
resources has an explicit license or written permission permitting redistribution and adaptation.
Public accessibility or an academic citation alone is insufficient. Each package resource
directory preserves its dataset-specific attribution and license; the repository BSD 3-Clause
license does not relicense included data.

Do not publish private delivery details, inaccessible local paths, checksums of unshared source
files, or preparation scripts that only reconstruct committed resources from private inputs.

## Resource and metadata policy

Every package-owned reference dataset follows `.llm/dataset_layout.md`:

- UTF-8 comma-delimited `X.csv` and `Y.csv` with headers;
- `metadata.json` with ordered labels, dimensions, public provenance, preparation, licensing,
  raw-resource hashes, and canonical-array hashes;
- a human-readable `README.md` describing direct raw-file access;
- a local `LICENSE.txt` containing the required attribution and redistribution terms;
- exactly one active matrix pair in the repository.

Metadata standardizes and validates the packaged resource. It does not create a public registry and
is not required when users supply their own `X` and `Y`.

## Public reconstruction exception

A reconstruction or preprocessing script is appropriate only when all of the following hold:

- the raw source is included or publicly obtainable;
- the transformation is scientifically relevant to how users should form `X` and `Y`;
- the code is intended to be read and run by programming users;
- the example exposes, rather than hides, the analytical choices.

No current reference dataset uses this exception. A future reconstruction requires a separate
licensing and ownership decision and remains dataset-specific.

## Separation of responsibilities

- **Programming user:** reads and prepares arbitrary `X` and `Y` from their own source.
- **Estimator API:** validates supplied model matrices and fits Pi-PLS.
- **Package-owned reference integration:** provides one canonical language-neutral resource set,
  named Python loading, public metadata, provenance, integrity checks, and licenses.
- **Examples:** use the named loader for package-owned reference data and show all later analytical
  choices; examples for user-owned data continue to form `X` and `Y` visibly.

## Prohibited directions

Do not introduce merely for examples:

- a public dataset registry or generic real-data loader;
- automatic downloading;
- runtime dependence on metadata sidecars for arbitrary user data;
- hidden helpers that conceal how user-owned `X` and `Y` were formed;
- private archive or local-path references in committed dataset materials;
- preparation-only scripts whose required source is unavailable to users;
- preprocessing learned across train/test or cross-validation boundaries.

## Analysis artifact transparency

Committed `X.csv` and `Y.csv` files are input assets, not a reason to serialize intermediate
analysis results. Every numbered real-data workflow keeps paths, selected rows, predictor-rank
profiles, validation reports, and inspection results in memory, creates figures directly, and
writes only final PDF outputs. Example 03 compares immutable Pi-PLS and ordinary-PLS paths; Pulp,
Sugarcane, and Tobacco expose the conditional predictor-rank profile at their selected component
count. Tobacco uses search-owned minimum-CV-MSE selection with a 10% relative tolerance and passes the returned count into
`predictor_rank_profile()`.

Generated PDFs are never committed or included in snapshots. Output-directory structure is
preserved through `.gitkeep` files.

## Publication boundary

Paper-specific data orchestration belongs in downstream reproduction repositories that depend on a
tagged `pipls` release. This repository's examples remain concise package-use examples and do not
grow into manuscript pipelines or complete comparison grids.

## Tutorial assets

The synthetic tutorial uses `make_pipls_train_test()` directly. Pulp tutorial renderers use
`load_pulp()` and record the packaged dataset identifier, provenance,
resource hashes, canonical-array hashes, repeated-CV protocol, OOF prediction count, selected pair,
evaluated predictor ranks, and generated-figure hashes in ignored manifests. Generated tutorial
assets are not alternative dataset representations.
