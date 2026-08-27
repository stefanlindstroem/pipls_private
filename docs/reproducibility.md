# Reproducibility

The `pipls` repository treats reproducibility as a software-product responsibility: distributed code,
numerical contracts, deterministic synthetic generation, transparent example data, executable examples, and transparent validation contracts should be reviewable from the
repository itself.

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

The executable real-data examples require Matplotlib. The annotated Pulp biplot uses `textalloc`
when available and otherwise retains the original Matplotlib label positions. The package-owned
loaders remove pandas from the numbered-example dependency set.
Install the dedicated extra with:

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
submodule imports, installed version-metadata agreement, a representative fixed Π-PLS fit and
prediction, and an explicit check that imports resolve from the temporary installation rather than
`src/`. Matplotlib and `textalloc` remain optional example dependencies and are not imported by the
runtime package.

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

The quick-start, synthetic-tutorial, Pulp-tutorial, and Home comparison figures are generated from
maintained repository calculations rather than committed image binaries. The synthetic tutorial
manifest records the generator configuration, selected rank pair, external-test provenance,
filenames, and SVG hashes. The Pulp tutorial manifest records dataset identity, version, source DOI,
license, package-resource and canonical-array hashes, evaluated ranks, boundary status, display
subsets, and the exhaustive search-domain SVG used by the path-and-selection guide. The Home
comparison manifest records the
[Pulp](datasets.md#pulp-real-data-integration) and
[Tobacco](datasets.md#tobacco-spectral-integration) dataset identities and versions,
the
[publication-default cross-covariance response-subspace policy](theory.md#response-subspace-selection),
the seeded five-fold validation protocol, case-specific predictor-rank search settings, component
and predictor-rank paths, both CV-MSE summaries, and the two SVG hashes. Every generated SVG contains one chart. Generated
directories are ignored by Git and removed by `make clean`.

The preview is served at `http://127.0.0.1:8000/` and updates as documentation files change. Stop it
with `Ctrl+C`. The stronger distribution check is:

```bash
make docs-dist
```

It builds a source distribution, unpacks it, creates a clean virtual environment, installs the
unpacked package with its documentation extra, and runs the same strict site build. This verifies
that `mkdocs.yml`, the Makefile, Markdown sources, JavaScript assets, generated-reference inputs,
package source, documentation renderers, their required example-support code, and the
[Pulp](datasets.md#pulp-real-data-integration) and
[Tobacco](datasets.md#tobacco-spectral-integration) data needed by the Home comparison are shipped
together. The clean build regenerates and
parses the declared SVG assets. Generated documentation assets and `site/` output are temporary and
are not part of the source distribution.

## Model-fitting reproducibility

`PiPLSRegression.random_state` accepts an integer seed, a NumPy `RandomState`, or `None`. The
default integer `0` is reproducible. `None` uses NumPy's global random state and should be chosen
only when repeatability is not required. The synthetic generator retains its separate explicit
integer-seed contract.

Centering and optional scaling are integral to `PiPLSRegression.fit`. The `scale` parameter remains
the default policy for both blocks, while `scale_x` and `scale_y` can override predictor and response
scaling independently. `PiPLSSearchCV` clones the complete fixed estimator or supported pipeline
inside every training fold, so each candidate learns statistics only from that fold. A subsequent
`search.refit(X, Y, ...)` call fits one selected rank pair on the complete supplied training set, so
centering and any enabled estimator or pipeline scaling are learned again from that full data.
Learned scaling must not be fitted globally before CV.

Record `response_subspace` as part of every model-fitting protocol. The package default
[`"cross_covariance"` response-subspace construction](theory.md#response-subspace-selection) is the
construction used in the [peer-reviewed companion publication](citation.md#companion-paper);
`"least_squares"` is a software extension outside that publication. `PiPLSSearchCV`
clones this setting unchanged and does not search over it automatically. When comparing the two
policies, materialize one validation protocol and reuse the same split indices for both searches so
that differences are not confounded with resampling variation. Record the policy together with
scaling, SVD configuration, rank-search policy, component-selection rule, and random seeds.

Maintained examples that use one ordinary five-fold regression partition specify
`KFold(n_splits=5, shuffle=True, random_state=0)` rather than relying on the package default. The
complete Pulp workflow instead uses ten seeded repetitions of five-fold CV. These choices avoid fold
assignments determined by file row order, and every stochastic maintained partition has an explicit
seed.

## Synthetic validation

`pipls.datasets` provides a deterministic latent-structure generator with known shared,
predictor-specific, and response-specific components. The
[Datasets and generators API](api/datasets.md#synthetic-generator) defines the Gaussian latent
distribution and seeded generator contract.

Scientific comparison studies, large simulation grids, paper-only comparators, and manuscript
figure or table orchestration remain downstream assets that pin a released package version. The
package test suite protects maintained numerical and API contracts directly.

## Reference datasets

[Pulp](datasets.md#pulp-real-data-integration),
[Sugarcane](datasets.md#sugarcane-spectral-integration), and
[Tobacco](datasets.md#tobacco-spectral-integration) are canonical package resources under
`src/pipls/_data/<dataset>/`. Each directory contains `X.csv`, `Y.csv`, `metadata.json`,
`README.md`, and `LICENSE.txt`; the same resources are included in wheels and source distributions.
The named loaders return these exact matrices without learned preprocessing.

The [dataset documentation](datasets.md) gives the original-source citation, DOI links, preparation,
license, dimensions, and direct raw-file locations for every integration. Every numbered
reference-data workflow uses the corresponding named loader, keeps component paths, selected rows,
predictor-rank profiles, validation reports, and inspection results in memory, and writes only its
final PDF figures.

Distribution validation checks both clean installed loading and the presence of all five files for
all three datasets. Package-resource tests verify declared raw-resource hashes, canonical `float64`
array hashes, and one active matrix pair per named dataset. Metadata supports provenance and review;
it is not required when users supply their own `X` and `Y`.

## Scope of reproducibility

The repository validates the installable package, its documented datasets, and its maintained
examples. Application-specific studies may add their own simulations, comparators,
tables, and figures without changing the package reproducibility contract.
