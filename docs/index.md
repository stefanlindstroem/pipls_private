# $\Pi$-PLS documentation

`pipls` is a Python package for $\Pi$-PLS, a PLS-family method for multivariate linear regression.
For routine modeling, $\Pi$-PLS can be used much like ordinary PLS: choose a component count, fit
and predict, inspect latent scores and loadings, examine regression coefficients, and assess
observed-versus-predicted values and residuals. The package exposes those familiar PLS-family
analysis quantities together with the additional $\Pi$-PLS-specific paired-direction
factorization.

## Why use $\Pi$-PLS?

In the problems examined in the [companion paper](citation.md#companion-paper), $\Pi$-PLS is
reported to be comparably robust to ordinary PLS while matching or improving its predictive
performance; in some settings, the predictive improvement is substantial. These are empirical
results for the studied problems rather than a guarantee that one method will be better for every
dataset.

The method differs from ordinary PLS in how it constructs the latent regression map. $\Pi$-PLS
represents that map through paired latent modes. Each retained mode contains one orthonormal
predictor direction, one orthonormal response direction, and one nonnegative dilation. This
structure adds method-specific interpretation without replacing the standard PLS-family analysis
workflow.

## Leakage-safe modeling and validation

The package provides one consistent workflow for fitting, model selection, prediction diagnostics,
and final refitting. Learned centering, scaling, and supported pipeline preprocessing are fitted
inside each cross-validation training fold rather than on the complete dataset before validation.
This keeps candidate evaluation fold-local and avoids preprocessing leakage across validation
boundaries.

The workflow also keeps different kinds of predictive evidence distinct:

- cross-validation is used to compare and select candidate models;
- selection-conditioned out-of-fold predictions can be inspected without refitting the folds;
- final refitting learns the selected model from the complete training data;
- nested cross-validation or an independent test set is used when an independent estimate of
  post-selection predictive performance is required.

These distinctions are carried explicitly by the search and inspection APIs so that training,
selection, diagnostic validation, and independent testing are not silently conflated. This follows
standard statistical practice for separating model development from independent performance
assessment.

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
the complete selection contract in a small deterministic problem: inspect the unselected component
path, choose a paired-mode count and create one selection, inspect the selected path and conditional
predictor rank, refit that exact row, and predict an independent test set.

Continue with [Pulp: a complete $\Pi$-PLS workflow](tutorials/pulp.md) for real-data loading,
selection-conditioned OOF inspection before final refitting, immutable inspection results, and
representative interpretation of standard PLS-family and $\Pi$-PLS-specific plots.

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
