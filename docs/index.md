# Pi-PLS documentation

Pi-PLS is a multivariate linear-regression method that represents the predictive relation through
paired latent modes. Each mode contains one orthonormal predictor direction, one orthonormal response
direction, and one nonnegative dilation.

## When Pi-PLS may be useful

Although Pi-PLS and ordinary PLS have similar names and belong to the same broad family of
latent-variable regression methods, their theoretical foundations differ. Pi-PLS diagonalizes its
latent regression map into paired predictor and response directions, giving each retained paired
mode a direct one-to-one predictor-response interpretation. It can also provide a more parsimonious
predictive model by attaining a given cross-validated mean squared error (CV-MSE) with fewer paired
latent modes.

Across a wide range of synthetic settings and real-world datasets examined during
development, Pi-PLS typically yields lower CV-MSE than ordinary PLS at a given number of paired latent modes (*cf*. example 04). This is not a general performance claim: no method is universally better.

## Quick start with Pulp

The installed package contains the multivariate Pulp dataset:

```python
from pipls import PiPLSSearchCV
from pipls.datasets import load_pulp

X, Y = load_pulp(return_X_y=True)
model = PiPLSSearchCV().fit(X, Y).refit(X, Y, rule="minimum_cv_mse")
Y_fitted = model.predict(X)
```

The first numbered example standardizes the eight response variables and combines all observed and
fitted values in one plot. This is a compact calibration-fit demonstration: `Y_fitted` comes from
the same observations used to fit the final model. Use a
[`oof_report()`](api/path.md#out-of-fold-report) for
selection-conditioned out-of-fold validation when the fitted search is retained.

## Choose a tutorial

After the quick start, [Inspect and select with synthetic data](tutorials/synthetic.md) introduces
the complete selection contract in a small deterministic problem: evaluate the component path,
choose a paired-mode count, inspect its conditionally selected predictor rank, fit one fixed model,
and predict an independent test set.

Continue with [Pulp: a complete Pi-PLS workflow](tutorials/pulp.md) for real-data loading,
fixed-parameter OOF predictions, immutable inspection results, and representative interpretation of
standard PLS-family and Pi-PLS-specific plots.

## Programming reference

- [API overview](api/index.md): generated signatures and method contracts.
- [Path selection](api/path.md): search parameters and immutable path results.
- [Path-selection details](path_analysis.md): bounds, policies, splitters, OOF output, and refitting.
- [Troubleshooting](troubleshooting.md): common fit, selection, scoring, and plotting problems.
- [Model inspection](model_inspection.md): general interpretation of fitted quantities and plots.
- [Examples](examples.md): maintained executable analyses and their outputs.

## Project validation

- [Reference datasets](datasets.md): provenance, licensing, adaptation, and matrix dimensions.
- [Reproducibility](reproducibility.md): software, data, and generated-documentation controls.
- [Compatibility](compatibility.md): supported Python and dependency versions.

## Scientific background

- [Theory](theory.md): the implemented matrix construction and rank interpretation.
- [Companion-manuscript synthetic data](manuscript_reproduction.md): generate the exact Gaussian
  latent distribution and distinguish distribution, seeded-dataset, and full-study reproduction.

## Project information

- [Authors, license, and citation](citation.md): copyright holders, commercial-use terms,
  dataset-license scope, and the companion paper.
