# Decision 0138: package-owned Pulp dataset loader

## Status

Accepted; Patch 2 implemented.

## Context

The Pulp dataset is already a licensed, documented repository asset and the principal real-data
Pi-PLS tutorial. Its current integration requires users and examples to locate
`datasets/pulp/X.csv` and `datasets/pulp/Y.csv` in a repository checkout and read them explicitly.
That transparent layout is appropriate for general repository datasets, but it prevents the
shortest installed-package demonstration from using the project's most relevant real multivariate
regression data.

Scikit-learn's small built-in datasets establish a familiar convention: a named loader returns a
structured dataset by default and can return the model matrices directly through
`return_X_y=True`. Pi-PLS already has the immutable `PiPLSDataset` container, so introducing a
`Bunch`, pandas requirement, network download, registry, or generic data-access framework would be
unnecessary.

The package remains unreleased at version `0.0.0`. Pulp can therefore move to one canonical
package-owned representation without compatibility aliases or duplicate active assets.

## Decision

Add one named loader in the dedicated dataset namespace:

```python
def load_pulp(
    *,
    return_X_y: bool = False,
) -> PiPLSDataset | tuple[FloatArray, FloatArray]:
    ...
```

`load_pulp()` mirrors the call style of `sklearn.datasets.load_linnerud()` but returns the existing
immutable `PiPLSDataset` by default. `return_X_y=True` returns the read-only `float64` predictor and
response arrays directly. The loader is exported from `pipls.datasets`, not from the top-level
`pipls` namespace.

The default result provides:

- `data` and `target` arrays with the canonical 46 observations, 14 predictors, and eight responses;
- ordered `feature_names` and `target_names` matching the active Pulp model matrices;
- stable sample identifiers `pulp-01` through `pulp-46`;
- public source, citation, version, and CC BY 4.0 provenance;
- recursively immutable metadata describing the analysis-facing selection and resource integrity.

The loader performs no network access, preprocessing, scaling, imputation, row filtering, or model
fitting. It has no `as_frame` parameter initially and introduces no pandas, PyYAML, or optional
runtime dependency. General users remain free to supply any array-like `X` and `Y` directly to the
estimators; `load_pulp()` is one named package example dataset, not a required registry layer or a
generic loading protocol.

The canonical active resources move to the installed package under:

```text
src/pipls/_data/pulp/
    X.csv
    Y.csv
    metadata.json
    README.md
    LICENSE.txt
```

Runtime loading uses `importlib.resources` and standard-library CSV/JSON handling so it works from a
wheel, an unpacked source distribution, and a source checkout. Package and distribution
configuration includes these resources explicitly. The JSON metadata replaces YAML only for the
package-owned runtime representation; Sugarcane and Tobacco continue to follow the repository
`X.csv`/`Y.csv`/`metadata.yaml` convention.

The former repository-facing Pulp files remain active only during the staged migration. Once every
maintained Pulp consumer uses `load_pulp()`, move the former layout without modification to:

```text
.llm/archive/pulp-repository-layout-v1/
```

That archive is development history only. Runtime code, examples, tests, served documentation,
wheels, and source distributions must not read or include it. The package resources are then the
sole active Pulp matrix representation. The archive may be removed in a later owner-authorized
cleanup after the loader contract has stabilized.

Implement the transition in five reviewable patches:

1. establish this decision and the guide-layer target;
2. add package resources, `load_pulp()`, loader tests, and installed-distribution validation while
   retaining the repository copy as a temporary parity source;
3. make Pulp the first compact fitted-value quick start without presenting in-sample fit as OOF
   validation;
4. migrate the complete Pulp example, tutorial renderer, tests, manifests, and documentation to the
   loader;
5. move the former repository layout into `.llm/archive/`, remove it from active dataset and
   distribution contracts, and run a final single-active-copy audit.

Public user documentation must describe only implemented stages. Patch 2 implements the package
resources, `load_pulp()`, parity tests, and clean-distribution smoke checks while retaining the
repository copy temporarily. The first-example and complete-workflow migrations remain assigned to
Patches 3 and 4, and archival remains assigned to Patch 5.

This decision supersedes Decisions 0019, 0062, and 0068 only where they require direct reading of
`datasets/pulp/X.csv` and `datasets/pulp/Y.csv` or prohibit a Pulp-specific package loader. Their
scientific column selection, provenance, workflow visibility, path inspection, validation,
interpretation, and rendering contracts otherwise remain in force. Decisions 0016 and 0018 remain
the general contract for user-owned real data and repository datasets; Pulp becomes one explicit
package-owned exception rather than a generic replacement.

## Consequences

- The shortest installed-package example can use real Pi-PLS data without a repository checkout.
- `PiPLSDataset.data` and `.target` gain a natural real-data use while plain arrays remain the
  primary estimator input.
- There is one named loader, not a registry or automatic downloader.
- Pulp provenance and licensing travel with installed distributions.
- Sugarcane and Tobacco retain transparent repository CSV ownership.
- The completed repository contains one active Pulp matrix location plus a temporary hidden
  historical archive excluded from runtime and distributions.
- No compatibility alias, duplicate active dataset, pandas return mode, or hidden preprocessing is
  introduced.
