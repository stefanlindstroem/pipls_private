# Pi-PLS documentation

Pi-PLS is a multivariate linear-regression method that represents the predictive relation through
paired predictor and response latent variables.

## Choose a tutorial

[First Pi-PLS model with synthetic data](tutorials/synthetic.md) is the recommended introduction. It
shows the complete selection contract in a small deterministic problem: evaluate the component
path, choose a component count, retrieve its conditionally selected predictor rank, fit one fixed
model, and predict an independent test set.

Continue with [Pulp: a complete Pi-PLS workflow](tutorials/pulp.md) for real-data loading,
fixed-parameter OOF predictions, immutable inspection results, and representative interpretation of
standard PLS-family and Pi-PLS-specific plots.

## Fit one known model directly

When `n_components` and `predictor_rank` are already known, use the fixed estimator directly:

```python
from pipls import PiPLSRegression

model = PiPLSRegression(
    n_components=2,
    predictor_rank=4,
).fit(X_train, Y_train)

Y_pred = model.predict(X_test)
```

This fit performs no parameter selection. See the
[fixed-regression reference](api/regression.md) for preprocessing, solver, fitted-state, and method
contracts.

## Programming reference

- [API overview](api/index.md): generated signatures and method contracts.
- [Path selection](api/path.md): search parameters and immutable path results.
- [Path-selection details](path_analysis.md): bounds, policies, splitters, OOF output, and refitting.
- [Troubleshooting](troubleshooting.md): common fit, selection, scoring, and plotting problems.
- [Model inspection](model_inspection.md): general interpretation of fitted quantities and plots.
- [Examples](examples.md): maintained executable analyses and their outputs.

## Project validation

- [Reference datasets](datasets.md): provenance, licensing, adaptation, and matrix dimensions.
- [Benchmarks](benchmarks.md): focused package-level numerical and selection checks.
- [Reproducibility](reproducibility.md): software, data, and generated-documentation controls.
- [Compatibility](compatibility.md): supported Python and dependency versions.

## Scientific background

- [Theory](theory.md): the implemented matrix construction and rank interpretation.
