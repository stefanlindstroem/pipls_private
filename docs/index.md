# Pi-PLS documentation

Pi-PLS is a multivariate linear-regression method that represents the predictive relation through
paired predictor and response latent variables.

## Start with the synthetic tutorial

[First Pi-PLS model with synthetic data](tutorials/synthetic.md) is the recommended introduction. It
shows the complete selection contract in a small deterministic problem: evaluate the component
path, choose a component count, retrieve its conditionally selected predictor rank, fit one fixed
model, and predict an independent test set.

Continue with [Pulp: a complete Pi-PLS workflow](tutorials/pulp.md) for real-data loading,
fixed-parameter OOF predictions, immutable inspection results, and scientific interpretation of
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

## Find a specific topic

- [Examples](examples.md): the maintained executable analyses and their outputs.
- [Path-selection API](api/path.md): generated signatures and result objects.
- [Advanced path-search behavior](path_analysis.md): bounds, policies, pipelines, and refitting.
- [Cross-validation](cross_validation.md): splitters, scoring, OOF output, and provenance.
- [Model inspection](model_inspection.md): general interpretation of fitted quantities and plots.
- [API overview](api/index.md): all generated public signatures and method contracts.
- [Compatibility](compatibility.md): supported Python and dependency versions.
- [Theory](theory.md): the implemented matrix construction and rank interpretation.
- [Reproducibility](reproducibility.md): data, software, and generated-documentation controls.
