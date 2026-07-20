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
