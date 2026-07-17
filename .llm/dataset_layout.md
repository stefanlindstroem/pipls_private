# Repository real-dataset layout

## Purpose

This contract standardizes committed analysis-facing real datasets without changing the public
estimator API. External programming users still read and prepare their own `X` and `Y`; they do not
need a metadata file or package loader.

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

Examples and reproduction scripts must read these files directly with ordinary NumPy or pandas
code. They must not call a package loader or parse `metadata.yaml` to construct the model matrices.

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
- `source`: public publication, repository, archive, or included-raw-data references;
- `license`: identifier, local license file when applicable, and redistribution status;
- `preparation`: transparent analysis-facing transformations from public source material to the
  committed model tables;
- `integrity`: SHA-256 hashes for the analysis files and relevant license assets.

The metadata may contain additional dataset-specific fields, but the required fields and meanings
must remain stable across datasets. A compact regular-axis descriptor must state the measured
quantity, axis name and unit, start, stop, step, ordering, and how CSV headers encode the axis; it
must define every predictor column without relying on hidden code.

## Public-facing provenance rule

Dataset assets must stand on their own for a programming user who sees only the public repository.
Do not include:

- personal delivery details or contributor-specific source paths;
- private archive names;
- inaccessible local filenames presented as upstream sources;
- checksums for source files that are not publicly available or included;
- references to internal preparation scripts.

When committed `X.csv` and `Y.csv` are adapted from public supplementary material, cite that public
material and describe the named selections or transformations directly. Development-only lineage
must not leak into repository metadata.

## Reconstruction and preprocessing code

Do not commit preparation-only code merely to document how private development files were split.
Public reconstruction code is allowed only when it operates on included or publicly obtainable raw
data and represents analysis-relevant work that users should see.

Corn is explicitly reserved for this second case: the raw data and the necessary reading and
preprocessing choices must be exposed transparently to programming users.

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

## Testing boundary

Repository tests verify that `X.csv`, `Y.csv`, and `metadata.yaml` exist and use supported technical
formats. They may check that the two CSV files are numeric, finite, non-empty, comma-delimited, and
row-aligned, and that the metadata file parses as a non-empty YAML mapping.

Tests do not duplicate the documentary contents of `metadata.yaml`: titles, descriptions,
citations, dimensions, variable lists, preparation prose, and recorded hashes remain reviewable
repository data rather than fixed test expectations. Exact scientific fixtures belong to Phase E4
or a paper-reproduction decision with explicit tolerances and update rules.
