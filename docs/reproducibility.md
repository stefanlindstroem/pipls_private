# Reproducibility

The `pipls` repository treats reproducibility as a software-product responsibility: distributed code,
numerical contracts, deterministic synthetic generation, transparent example data, executable
examples, and lightweight validation benchmarks should be reviewable from the repository itself.

## Software reproducibility

A clean source checkout should support:

```bash
python -m pip install ".[dev]"
make check
make examples
make build
make dist-check
```

Tests cover the numerical core, estimator API, model selection, cross-validation boundaries,
scikit-learn compatibility, dataset readability, example helper/artifact contracts, and repository
structure. `make examples` is the separate application-validation target and runs every numbered
example, including the complete real-data analyses. Randomized numerical paths and synthetic
generators expose explicit random-state controls.

The executable real-data examples require pandas and Matplotlib; the annotated Pulp biplot also
uses `adjustText`. Install their dedicated extra with:

```bash
python -m pip install ".[examples]"
make examples
```

The development extra already includes these packages. Re-run `python -m pip install ".[dev]"`
after pulling dependency changes into an existing validation environment. Contributors who
need an editable checkout use the setup documented in the contributor guide.

## Installed-distribution reproducibility

The stronger packaging check is:

```bash
make dist-check
```

It builds the wheel and source distribution once in a temporary directory, creates a separate clean
virtual environment for each artifact, and installs each exact artifact path while running outside
the repository checkout. Both environments execute the same smoke test: public package and
submodule imports, installed version-metadata agreement, a representative fixed Pi-PLS fit and
prediction, and an explicit check that imports resolve from the temporary installation rather than
`src/`. Matplotlib and `adjustText` remain optional example dependencies and are not imported by
the runtime package.

The temporary environments and artifacts are removed after the check. This target validates
installation behavior; `make check` remains the ordinary source-checkout test suite, and
`make docs-dist` separately validates the distributed documentation inputs.

## Documentation reproducibility

Install the dedicated documentation dependencies and build or preview the site from a checkout:

```bash
python -m pip install ".[docs]"
make docs-figures
make docs
make docs-serve
```

The synthetic tutorial figures are generated from the same deterministic calculation shown in
example 02. Its manifest records the generator configuration, selected rank pair, external-test
provenance, filenames, and SVG hashes. The Pulp figures are generated from example 05's direct
calculation; their manifest additionally records dataset hashes, evaluated ranks, boundary status,
and display subsets. Every SVG contains one chart. Generated directories are ignored by Git and
removed by `make clean`.

The preview is served at `http://127.0.0.1:8000/` and updates as documentation files change. Stop it
with `Ctrl+C`. The stronger distribution check is:

```bash
make docs-dist
```

It builds a source distribution, unpacks it, creates a clean virtual environment, installs the
unpacked package with its documentation extra, and runs the same strict site build. This verifies
that `mkdocs.yml`, the Makefile, Markdown sources, JavaScript assets, generated-reference inputs,
package source, synthetic and Pulp tutorial examples, Pulp data, and both renderers are shipped
together. The clean build regenerates and parses the declared SVG assets. Generated tutorial assets and `site/` output
are temporary and are not part of the source distribution.

## Model-fitting reproducibility

`PiPLSRegression.random_state` accepts an integer seed, a NumPy `RandomState`, or `None`. The
default integer `0` is reproducible. `None` uses NumPy's global random state and should be chosen
only when repeatability is not required. The synthetic generators retain their separate explicit
integer-seed contract.

Centering and optional scaling are integral to `PiPLSRegression.fit`. `PiPLSSearchCV` clones fixed
estimators inside every training fold, so each candidate learns statistics only from that fold.
With `refit=True`, the pair chosen by `selection_rule` learns them again from the complete supplied
training set. Learned scaling must not be fitted globally before CV.

## Synthetic validation

`pipls.datasets` provides deterministic latent-structure generators with known shared,
predictor-specific, and response-specific components. The separate
[companion-manuscript synthetic-data guide](manuscript_reproduction.md) documents the exact
Gaussian latent distribution and distinguishes reproducing that distribution from reproducing one
seeded realization or a complete publication study.

The package benchmark plan is documented in [`benchmarks.md`](benchmarks.md). Each benchmark answers
one question and writes one minimal CSV output; unrelated metrics, software metadata, and timings
are not combined into a universal table. Publication grids, paper-only comparators, and manuscript
figure or table orchestration remain downstream assets that pin a released package version.

## Reference datasets

Each committed real dataset uses `X.csv`, `Y.csv`, and documentary `metadata.yaml`, together with
public provenance and redistribution terms. Current integrations are:

- pulp;
- sugarcane LabSpec spectroscopy;
- tobacco FT-NIR spectroscopy.

The [dataset documentation](datasets.md) gives the original-source citation and resolvable DOI link
for every integration, together with related data papers where applicable.

Examples read `X.csv` and `Y.csv` directly. Example 04 operates on immutable Pi-PLS and ordinary-PLS
paths in memory. Pulp, Sugarcane, and Tobacco operate on `component_path_`, scikit-learn OOF
predictions, and immutable inspection results in memory. Every numbered real-data workflow writes only
final PDF figures; generated CSV products are reserved for the focused benchmark scripts.
Metadata supports repository review but is not required by the runtime API. Dataset-specific
transformations that matter to users are described publicly; private preparation archives and
inaccessible paths are not part of the repository.

## Scope of reproducibility

The repository validates the installable package, its documented datasets, and its maintained
examples and benchmarks. Application-specific studies may add their own simulations, comparators,
tables, and figures without changing the package reproducibility contract.
