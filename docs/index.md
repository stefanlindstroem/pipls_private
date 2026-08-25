# Π-PLS documentation

`pipls` is a Python package for the multivariate linear regression model panoramic partial least squares (Π-PLS), where Π stands for the Greek πανοραμικός (“panoramic”).
For routine modeling, Π-PLS can be used much like ordinary PLS: choose a component count, fit
and predict, inspect latent scores and loadings, examine regression coefficients, and assess
observed-versus-predicted values and residuals. The package exposes those familiar PLS-family
analysis quantities together with the additional Π-PLS-specific paired-direction
factorization.

## Why use Π-PLS?

In the problems examined in the [companion paper](citation.md#companion-paper), Π-PLS is
reported to be comparably robust to ordinary PLS while matching or improving its predictive
performance; the predictive improvement is substantial in some settings. These are empirical
results which may not hold for every dataset.

**Fewer shared components can be sufficient.** For many datasets, including the
[Pulp](datasets.md#pulp-real-data-integration) and
[Tobacco](datasets.md#tobacco-spectral-integration) datasets available through this package, Π-PLS
reaches a low cross-validated prediction-error region with fewer shared components than ordinary
PLS, as seen in the following figure:

<div class="grid" markdown>
![Pulp component-path comparison between Π-PLS and ordinary PLS](assets/generated/home/pulp_component_parsimony.svg)

![Tobacco component-path comparison between Π-PLS and ordinary PLS](assets/generated/home/tobacco_component_parsimony.svg)
</div>

Both panels use the same seeded five-fold validation splits for Π-PLS and ordinary PLS, and the error bars show the split-to-split standard deviation. 

**Π-PLS gives a one-to-one relation between predictor and response directions.** The Π-PLS method differs from ordinary PLS in how it constructs the latent regression map. Π-PLS
represents that map through [paired latent modes](theory.md#diagonal-latent-coupling). Each retained
mode contains one orthonormal predictor direction, one orthonormal response direction, and one
nonnegative dilation. This structure adds method-specific interpretation without replacing the
standard PLS-family analysis workflow.

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

## Quick start with Pulp dataset

The installed package contains the multivariate
[Pulp dataset](datasets.md#pulp-real-data-integration), for which a Π-PLS multivariate regeression
model is cretad in one line of code:

```python
from pipls import PiPLSSearchCV
from pipls.datasets import load_pulp

X, Y = load_pulp(return_X_y=True)
model = PiPLSSearchCV().fit(X, Y).refit(X, Y, rule="minimum_cv_mse")
Y_fitted = model.predict(X)
```

This is a compact calibration-fit demonstration: `Y_fitted` comes from
the same observations used to fit the final model. Study the complete [quick start tutorial](tutorials/quick_start.md) with prediction diagnostics.

## Choose a tutorial

After the quick start, [Inspect and select with synthetic data](tutorials/synthetic.md) introduces
the complete selection contract in a small deterministic problem: inspect the unselected component
path, choose a paired-mode count and create one selection, inspect the selected path and conditional
predictor rank, refit that exact row, and predict an independent test set.

Continue with [Pulp: a complete Π-PLS workflow](tutorials/pulp.md) for real-data loading,
selection-conditioned out-of-fold (OOF) inspection before final refitting, immutable inspection results, and
representative interpretation of standard PLS-family plots and Π-PLS-specific plots.

## Programming reference

- [Overview](api/index.md): public objects and canonical reference destinations.
- [`PiPLSRegression`](api/regression.md): fixed-estimator parameters and fitted results.
- [`PiPLSSearchCV`](api/path.md): search-estimator parameters and post-fit lifecycle.
- [Path and selection](path_selection.md): search bounds, rank policies, scoring, CV, path evidence, and selection rules.
- [OOF diagnostics](oof_diagnostics.md): stored-split reuse, ordered OOF predictions, coverage, and selection-conditioned interpretation.
- [Model inspection](model_inspection.md): numerical fitted-model quantities and interpretation.
- [Datasets and generators](api/datasets.md): packaged data, synthetic generators, and truth records.
- [Troubleshooting](troubleshooting.md): warnings, validation failures, lifecycle errors, and expensive searches.

## Project validation

- [Reference datasets](datasets.md): provenance, licensing, adaptation, and matrix dimensions.
- [Reproducibility](reproducibility.md): software, data, and generated-documentation controls.
- [Compatibility](compatibility.md): supported Python and dependency versions.

## Scientific background

- [Theory](theory.md): the implemented matrix construction, response-subspace policies, and rank interpretation.
- [Companion-manuscript synthetic data](manuscript_reproduction.md): generate the exact Gaussian
  latent distribution and distinguish distribution, seeded-dataset, and full-study reproduction.

## Project information

- [Authors, license, and citation](citation.md): copyright holders, commercial-use terms,
  dataset-license scope, and the [companion paper](citation.md#companion-paper).
