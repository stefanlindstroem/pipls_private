# Repository real-dataset layout

## Purpose

This contract standardizes committed analysis-facing repository datasets without changing the
public estimator API. External programming users still read and prepare their own `X` and `Y`; they
do not need a metadata file or package loader. Decision 0138 authorizes one staged package-owned
Pulp exception; until that migration completes, Pulp remains governed by this current layout.

## Accepted Pulp exception and migration state

Decision 0138 will move the canonical active Pulp assets to
`src/pipls/_data/pulp/` and expose them through `pipls.datasets.load_pulp()`. The package-owned
representation uses `metadata.json` so runtime loading requires only the standard library. This does
not alter the repository layout for Sugarcane or Tobacco and does not create a generic dataset
registry.

During Patches 2--4, `datasets/pulp/` remains temporarily as an exact parity source while package
resources and maintained consumers are validated. Patch 5 moves that former layout without
modification to `.llm/archive/pulp-repository-layout-v1/`. The archive is excluded from runtime,
served documentation, wheels, source distributions, and active dataset tests. After that cleanup,
this repository-layout contract applies to Sugarcane and Tobacco, while the package-resource
contract in Decision 0138 applies to Pulp.

The present Patch 1 state changes no files or consumers below `datasets/pulp/`; all current layout
requirements remain executable until the later patches complete the transition.

## Required files

Every committed real dataset directory under `datasets/<dataset-id>/` must contain:

- `X.csv`: predictor matrix;
- `Y.csv`: response matrix;
- `metadata.yaml`: public repository description and provenance.

Additional human-readable files such as `README.md` and license files may be included when needed.
Do not add internal conversion scripts or references to private development inputs.

## CSV contract

`X.csv` and `Y.csv` must:

- be UTF-8 text;
- use a comma as delimiter;
- contain one header row;
- contain only analysis-facing numeric model columns;
- preserve a documented common row order;
- use the exact filenames `X.csv` and `Y.csv`.

Examples must read these files directly with ordinary NumPy or pandas code. They must not call a
package loader or parse `metadata.yaml` to construct the model matrices.

## Metadata contract

`metadata.yaml` is required for repository-included datasets but is not a runtime input. It must
use `schema_version: 1` and contain these top-level fields:

- `dataset`: stable identifier, title, version, and summary;
- `files`: `predictors: X.csv` and `responses: Y.csv`;
- `format`: CSV type, comma delimiter, UTF-8 encoding, and header status;
- `dimensions`: sample, predictor, and response counts;
- `predictors`: ordered names and non-empty descriptions, or an exact compact descriptor for a
  regular high-dimensional axis such as a wavelength grid;
- `responses`: ordered names and non-empty descriptions;
- `sample_alignment`: row-alignment method and explanation;
- `missing_values`: declared policy for predictors and responses;
- `source`: public publication, repository, archive, or included-raw-data references, including
  DOI values and resolvable DOI URLs when available;
- `license`: identifier, local license file when applicable, and redistribution status;
- `preparation`: transparent analysis-facing transformations from public source material to the
  committed model tables;
- `integrity`: SHA-256 hashes for the analysis files and relevant license assets.

The metadata may contain additional dataset-specific fields, but the required fields and meanings
must remain stable across datasets. A compact regular-axis descriptor must state the measured
quantity, axis name and unit, start, stop, step, ordering, and how CSV headers encode the axis; it
must define every predictor column without relying on hidden code.

## Source-level licensing gate

Before files are added under `datasets/`, verify that the exact source material used to derive them
has an explicit license or written permission granting redistribution and adaptation for general
use. A public download page, a research-use statement, or a license attached to a different cleaned
or reduced derivative is not enough.

Record the source license, attribution requirements, and redistribution conclusion in
`metadata.yaml`, and include the governing license or permission text locally when appropriate. If
the source-level grant is absent or ambiguous, do not commit the source, derived matrices, or an
automatic downloader for them.

## Public-facing provenance rule

Dataset assets must stand on their own for a programming user who sees only the public repository.
Do not include:

- personal delivery details or contributor-specific source paths;
- private archive names;
- inaccessible local filenames presented as upstream sources;
- checksums for source files that are not publicly available or included;
- references to internal preparation scripts.

When committed `X.csv` and `Y.csv` are adapted from public supplementary material, cite that public
material and describe the named selections or transformations directly. The served dataset
documentation must reproduce the original-source citation and provide a resolvable DOI link for
each integration; related publications are identified separately and must not replace the original
data source. Development-only lineage must not leak into repository metadata.

## Reconstruction and preprocessing code

Do not commit preparation-only code merely to document how private development files were split.
Public reconstruction code is allowed only when it operates on included or publicly obtainable raw
data and represents analysis-relevant work that users should see.

No current dataset uses this exception. Any future reconstruction must operate on source material
whose redistribution and adaptation rights have already been verified.

## Separation from the programming-user contract

Repository consistency does not imply a public data-management framework. The ordinary user
workflow remains:

```python
X = ...
Y = ...
model = PiPLSRegression(n_components=2, predictor_rank=2).fit(X, Y)
```

A user may ignore `metadata.yaml`, use different filenames, use another delimiter, read a database,
or obtain arrays from any domain-specific source. The repository convention exists for review,
reproducibility, and consistent examples only.

## Testing boundary

Repository tests verify that `X.csv`, `Y.csv`, and `metadata.yaml` exist and use supported technical
formats. They may check that the two CSV files are numeric, finite, non-empty, comma-delimited, and
row-aligned, and that the metadata file parses as a non-empty YAML mapping.

Tests do not duplicate the documentary contents of `metadata.yaml`: titles, descriptions,
citations, dimensions, variable lists, preparation prose, and recorded hashes remain reviewable
repository data rather than fixed test expectations. Exact scientific fixtures require an explicit decision with scientific meaning, tolerances, and
update rules.
