# Reproducibility

The `pipls` repository treats reproducibility as a software-product responsibility: released code,
numerical contracts, deterministic synthetic generation, transparent example data, executable
examples, and lightweight validation benchmarks should be reviewable from the repository itself.

## Software reproducibility

A clean source checkout should support:

```bash
python -m pip install -e ".[dev]"
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

The executable real-data examples require pandas and Matplotlib. Install their dedicated extra with:

```bash
python -m pip install -e ".[examples]"
make examples
```

The development extra already includes these packages. Re-run `python -m pip install -e ".[dev]"`
after pulling dependency changes into an existing virtual environment.

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
`src/`. The check imports `pipls.plotting` but does not call optional plotting functions, so it also
protects the plotting module's import-time independence from Matplotlib.

The temporary environments and artifacts are removed after the check. This target validates
installation behavior; `make check` remains the ordinary source-checkout test suite, and
`make docs-dist` separately validates the distributed documentation inputs.

## Documentation reproducibility

Install the dedicated documentation dependencies and build or preview the site from a checkout:

```bash
python -m pip install -e ".[docs]"
make docs-figures
make docs
make docs-serve
```

The Pulp tutorial figures are generated from the canonical executable workflow before the strict
build or preview starts. Each SVG contains one chart, while `manifest.json` records the dataset
hashes, selected rank pair, display subset, prediction provenance, filenames, and SVG hashes. The
generated directory is ignored by Git and removed by `make clean`.

The preview is served at `http://127.0.0.1:8000/` and updates as documentation files change. Stop it
with `Ctrl+C`. The stronger distribution check is:

```bash
make docs-dist
```

It builds a source distribution, unpacks it, creates a clean virtual environment, installs the
unpacked package with its documentation extra, and runs the same strict site build. This verifies
that `mkdocs.yml`, the Makefile, Markdown sources, JavaScript assets, generated-reference inputs,
package source, Pulp data, canonical workflow, and tutorial renderer are shipped together. The clean
build regenerates and parses the declared SVG assets. Generated tutorial assets and `site/` output
are temporary and are not part of the source distribution.

## Model-fitting reproducibility

`PiPLSRegression.random_state` accepts an integer seed, a NumPy `RandomState`, or `None`. The
default integer `0` is reproducible. `None` uses NumPy's global random state and should be chosen
only when repeatability is not required. The synthetic generators retain their separate explicit
integer-seed contract.

Centering and optional scaling are integral to `PiPLSRegression.fit`. `PiPLSPathCV` clones fixed
estimators inside every training fold, so each candidate learns statistics only from that fold.
With `refit=True`, the chosen fixed pair learns them again from the complete supplied training set.
Learned scaling must not be fitted globally before CV.

## Synthetic validation

`pipls.datasets` provides deterministic latent-structure generators with known shared,
predictor-specific, and response-specific components. The package benchmark plan is documented in
[`benchmarks.md`](benchmarks.md). Each benchmark answers one question and writes one minimal CSV
output; unrelated metrics, software metadata, and timings are not combined into a universal table.

## Reference datasets

Each committed real dataset uses `X.csv`, `Y.csv`, and documentary `metadata.yaml`, together with
public provenance and redistribution terms. Current integrations are:

- pulp;
- sugarcane LabSpec spectroscopy;
- tobacco FT-NIR spectroscopy.

Examples read `X.csv` and `Y.csv` directly. Their comparison and plotting helpers are imported
Python functions, while the generated CSV files remain the artifact boundary used by the PDF.
Metadata supports repository review but is not required by the runtime API. Dataset-specific
transformations that matter to users are described publicly; private preparation archives and
inaccessible paths are not part of the repository.

## Scope of reproducibility

The repository validates the installable package, its documented datasets, and its maintained
examples and benchmarks. Application-specific studies may add their own simulations, comparators,
tables, and figures without changing the package reproducibility contract.
