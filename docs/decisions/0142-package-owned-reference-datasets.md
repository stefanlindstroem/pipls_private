# Decision 0142: package-owned reference datasets

## Status

Accepted; implementation in progress.

## Context

Decision 0138 made Pulp the first named dataset distributed with Pi-PLS. The resulting
`load_pulp()` workflow is useful in an installed package, preserves immutable labels and provenance,
and avoids a repository-relative file dependency. Sugarcane and Tobacco now have the same relevant
properties: they are licensed, curated reference analyses with fixed predictor and response
matrices, documented scientific transformations, and maintained Pi-PLS examples.

Keeping Sugarcane and Tobacco only under `datasets/` creates two avoidable inconsistencies. Their
examples require a repository checkout and pandas merely to read fixed CSV resources, while Pulp
works from an installed wheel. Their analysis matrices are also less discoverable through the
Python API even though they are recurring reference datasets with verified redistribution rights.

Moving the matrices into package resources must not make them Python-only or obscure them from R,
C++, MATLAB, Julia, or other programming users. The package resources are ordinary text files in a
wheel, source distribution, or source checkout. Their canonical locations and accompanying
metadata, README, and license files therefore need an explicit language-neutral documentation
contract.

The package remains unreleased at version `0.0.0`. The transition can end with one active matrix
copy per dataset and no compatibility aliases, generic registry, or duplicate repository layout.

## Decision

Pi-PLS owns three named reference-dataset loaders in `pipls.datasets`:

```python
def load_pulp(
    *,
    return_X_y: bool = False,
) -> PiPLSDataset | tuple[FloatArray, FloatArray]:
    ...


def load_sugarcane(
    *,
    return_X_y: bool = False,
) -> PiPLSDataset | tuple[FloatArray, FloatArray]:
    ...


def load_tobacco(
    *,
    return_X_y: bool = False,
) -> PiPLSDataset | tuple[FloatArray, FloatArray]:
    ...
```

All three loaders share one narrow contract:

- the default result is the existing immutable `PiPLSDataset`;
- `return_X_y=True` returns fresh read-only `float64` predictor and response arrays;
- loaders are exported from `pipls.datasets`, not from the top-level `pipls` namespace;
- loading performs no network access, filtering, imputation, centering, scaling, spectral
  preprocessing, or model fitting;
- runtime parsing uses `importlib.resources` plus standard-library CSV and JSON support;
- no pandas, PyYAML, optional data extra, `as_frame` mode, downloader, registry, or generic
  `load_dataset(name)` interface is introduced.

The canonical active resources are:

```text
src/pipls/_data/
    pulp/
        X.csv
        Y.csv
        metadata.json
        README.md
        LICENSE.txt
    sugarcane/
        X.csv
        Y.csv
        metadata.json
        README.md
        LICENSE.txt
    tobacco/
        X.csv
        Y.csv
        metadata.json
        README.md
        LICENSE.txt
```

Each `X.csv` and `Y.csv` pair is the exact analysis-facing matrix pair returned by its loader.
Metadata records ordered labels, source-level transformations, row alignment, public provenance,
licensing, raw-resource hashes, and canonical `float64` array hashes. The package-level sample
identifiers are stable identifiers for the returned row order:

- `pulp-01` through `pulp-46`;
- `sugarcane-01` through `sugarcane-57`;
- `tobacco-001` through `tobacco-347`.

These are package identifiers, not assertions that the upstream sources use the same identifiers.
Source sample numbers, exclusions, and row-alignment information remain documentary metadata.

The resource directories are intentionally language-neutral. Public dataset documentation must
list the canonical paths and explain that the same CSV, JSON, README, and license files are
available:

- in a source checkout or unpacked source distribution under `src/pipls/_data/<dataset>/`;
- inside a wheel, which is a ZIP archive, under `pipls/_data/<dataset>/`;
- in an installed environment under the environment-specific
  `<site-packages>/pipls/_data/<dataset>/` location.

For reproducible non-Python use, documentation should recommend a tagged source release, source
distribution, or wheel rather than a moving development branch or an environment-specific installed
path. Each resource README states that the files may be used independently of Python and identifies
the local matrix, metadata, provenance, and license resources.

At the end of the transition there is one active matrix representation for each named dataset. The
current `datasets/sugarcane/` and `datasets/tobacco/` copies may remain only as temporary parity
sources while their package resources and consumers are migrated. They are then removed rather than
archived as duplicate active data. The existing hidden Pulp archive remains excluded development
history and is not restored as an active data source.

Implement the transition in six reviewable patches:

1. establish this decision and the guide-layer target;
2. generalize the current Pulp-specific private resource loader without changing public behavior;
3. add Sugarcane resources, `load_sugarcane()`, integrity tests, and clean-distribution validation
   while retaining the repository copy as a temporary parity source;
4. add Tobacco resources, `load_tobacco()`, integrity tests, and clean-distribution validation under
   the same temporary parity rule;
5. migrate every maintained Sugarcane and Tobacco consumer to the named loaders and update active
   API, dataset, example, and maintainer documentation;
6. remove the duplicate repository matrices, publish the language-neutral raw-file locations,
   enforce one active resource pair per dataset, and mark this decision implemented.

## Implementation status

Patches 1 through 4 are implemented. The decision and guide-layer target are established, and the
shared private machinery handles resource access, metadata and CSV parsing, shape validation,
resource and canonical-array integrity checks, provenance, and stable sample identifiers.
`load_pulp()` preserves its original public contract. `load_sugarcane()` exposes the immutable
57 by 1,721 LabSpec matrix and four responses, and `load_tobacco()` exposes the immutable 347 by
1,557 raw FT-NIR matrix and 13 responses. Both spectral package resources have exact matrix parity
with their temporary repository copies and pass clean wheel/source-distribution loading. The
Tobacco attribution file is corrected before packaging so the canonical and temporary copies both
identify the Tobacco source. Maintained Sugarcane and Tobacco consumers continue to use the
repository copies until Patch 5.

This decision refines Decision 0138 from one package-owned Pulp exception to a closed set of three
maintained reference datasets. It supersedes Decisions 0022, 0023, 0067, 0069, and 0070 only where
they require direct reading from `datasets/sugarcane/` or `datasets/tobacco/`. Their scientific
matrix definitions, provenance, selection, validation, interpretation, and rendering contracts
otherwise remain in force. Decisions 0016 and 0018 continue to govern user-owned real data and any
repository dataset not explicitly assigned package ownership.

## Consequences

- Pulp, Sugarcane, and Tobacco can be loaded from a clean installed package through one consistent
  named-loader vocabulary.
- Their canonical CSV matrices remain directly usable outside Python and are documented as such.
- Examples 04, 06, and 07 can stop depending on repository-relative paths and pandas for fixed
  dataset ingestion.
- Package distributions grow because the spectral matrices travel with the base installation.
- Resource hashes, canonical-array hashes, installed-distribution loading, and single-copy location
  become durable tests for all three datasets.
- The package still owns no general real-data ingestion framework: arbitrary users continue to
  supply `X` and `Y` directly.
- No generic registry, downloader, compatibility alias, pandas return mode, hidden preprocessing,
  or duplicate active matrix representation is introduced.
