# Reference-dataset resource layout

## Purpose

This contract defines the active layout for package-owned reference datasets. It keeps the Python
loader representation and the language-neutral raw-file representation identical and prevents
duplicate active matrix copies.

Decision 0142 established Pulp. Decision 0142 extends the same final contract to Sugarcane and
Tobacco.

## Canonical active layout

The repository contains exactly three active reference-dataset directories:

```text
src/pipls/_data/
    pulp/
    sugarcane/
    tobacco/
```

Each directory contains exactly:

```text
X.csv
Y.csv
metadata.json
README.md
LICENSE.txt
```

There is no active top-level `datasets/` directory. Historical material below `.llm/archive/` is
excluded development history and is never a runtime, test-fixture, documentation, wheel, or source-
distribution input.

## Matrix files

- `X.csv` is UTF-8, comma-delimited, has one non-empty unique header per predictor, and contains
  finite numeric values.
- `Y.csv` follows the same technical format for responses.
- `X.csv` and `Y.csv` have the same row count.
- Their row and column orders are the analysis-facing orders returned by the named loader.
- Loading applies no hidden preprocessing.

## Metadata

`metadata.json` is the package-owned documentary and integrity manifest. It records:

- schema version, dataset identifier, title, version, and summary;
- matrix dimensions;
- ordered feature and response labels;
- variable and physical-axis descriptions where relevant;
- source-row alignment, exclusions, and missing-value policy;
- public source citations, DOI URLs, preparation, and license;
- SHA-256 hashes of the raw resources;
- SHA-256 hashes of canonical little-endian C-order `float64` arrays.

Runtime loading verifies the declared resource and array hashes. Public tests should protect the
schema and numerical integrity, not freeze incidental narrative wording.

## Human and language-neutral access

`README.md` identifies the local files, provenance, transformations, and direct non-Python access
locations. `LICENSE.txt` preserves the governing dataset attribution and redistribution terms.

Public documentation lists each complete source-checkout path under
`src/pipls/_data/<dataset>/`, the wheel location under `pipls/_data/<dataset>/`, and the normal
installed location below `<site-packages>/pipls/_data/<dataset>/`. It recommends tagged source
releases, source distributions, or wheels for reproducible external use.

## Python loading

The named loaders under `pipls.datasets` are the only package-owned loading API:

```python
load_pulp()
load_sugarcane()
load_tobacco()
```

They use `importlib.resources` and standard-library CSV/JSON parsing. No generic registry,
downloader, `as_frame` mode, or pandas/PyYAML runtime dependency follows from this layout.

## Single-copy and distribution contract

Tests verify:

- exactly one active `X.csv` and `Y.csv` pair per named dataset;
- all five files are present in the source tree, wheel, and source distribution;
- raw-resource and canonical-array hashes match `metadata.json`;
- loaders work from isolated wheel and source-distribution installations;
- `.llm/archive/` is absent from distributions;
- the public dataset guide contains source DOI links and raw-resource locations.

## Future datasets

A fourth package-owned reference dataset requires an explicit owner decision extending the closed
named-loader set and must satisfy the same licensing, provenance, single-copy, distribution, and
raw-access contracts.

A future repository-only or user-owned example dataset is not implied by this layout. If such an
asset is proposed, its ownership and file convention require a separate decision; arbitrary users
remain free to use any file organization before supplying `X` and `Y`.
