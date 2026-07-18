# Reproducibility

The `pipls` repository treats reproducibility as a software-product responsibility: released code,
numerical contracts, deterministic synthetic generation, transparent example data, executable
examples, and lightweight validation benchmarks should be reviewable independently of any one
scientific paper.

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
generators expose explicit random seeds.

The executable real-data examples require pandas and Matplotlib. Install their dedicated extra with:

```bash
python -m pip install -e ".[examples]"
make examples
```

The development extra already includes these packages. Re-run `python -m pip install -e ".[dev]"`
after pulling dependency changes into an existing virtual environment.

## Model-fitting reproducibility

Centering and optional scaling are integral to `PiPLSRegression.fit`. During cross-validation,
statistics are learned only from each training fold. After selection, the chosen model is refitted
on the complete supplied training set. Learned scaling must not be fitted globally before CV.

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

Examples read `X.csv` and `Y.csv` directly. Metadata supports repository review but is not required
by the runtime API. Dataset-specific transformations that matter to users are described publicly;
private preparation archives and inaccessible paths are not part of the repository.

## Publication reproduction

A publication may depend on a tagged `pipls` release and maintain its own complete simulations,
comparators, cached results, tables, and figure-generation environment. Such downstream artifacts
are separate from package validation and do not determine this repository's public surface.
