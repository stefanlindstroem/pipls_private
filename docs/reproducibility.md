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
make build
```

Tests cover the numerical core, estimator API, model selection, cross-validation boundaries,
scikit-learn compatibility, datasets, examples, and repository structure. Randomized numerical
paths and synthetic generators expose explicit random seeds.

## Model-fitting reproducibility

Centering and optional scaling are integral to `PiPLSRegression.fit`. During cross-validation,
statistics are learned only from each training fold. After selection, the chosen model is refitted
on the complete supplied training set. Learned scaling must not be fitted globally before CV.

## Synthetic validation

`pipls.datasets` provides deterministic latent-structure generators with known shared,
predictor-specific, and response-specific components. The accepted version-1 package benchmark
contract is documented in [`benchmarks.md`](benchmarks.md) and stored under `benchmarks/`. It
defines prediction, rank-selection, subspace, numerical-consistency, and resource metrics without
freezing publication claims or broad result files.

## Reference datasets

Each committed real dataset uses `X.csv`, `Y.csv`, and documentary `metadata.yaml`, together with
public provenance and redistribution terms. Current integrations are:

- Linnerud;
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
