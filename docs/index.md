# Pi-PLS documentation

Pi-PLS is a multivariate linear-regression method for predicting one or more response variables
from a block of predictors. It represents the predictive relation through a small number of paired
predictor and response latent variables.

## What Pi-PLS does

Two parameters control model complexity:

- `n_components` is the number of paired latent variables used for prediction;
- `predictor_rank` is the dimension of the predictor subspace retained before those pairs are
  estimated.

In routine use, `PiPLSPathCV` evaluates successive component counts by cross-validation. For each
count it selects a predictor rank and reports cross-validated mean squared error (CV-MSE). Plotting
CV-MSE against the number of components gives the model-selection curve used throughout the
examples. A common choice is the
elbow or plateau where additional components give little improvement, rather than automatically
using the absolute minimum.

See the [theory overview](theory.md) for the mathematical construction and the distinct roles of
the two ranks.

## Start here

1. [`quickstart.md`](quickstart.md): literal NumPy matrices, one fixed fit, predictions, and one
   latent-factor plot;
2. [`parameter_selection.md`](parameter_selection.md): scan component counts, inspect CV-MSE, and
   choose the final fixed model;
3. [`examples.md`](examples.md): synthetic, comparison, and complete real-data workflows;
4. [`estimator_api.md`](estimator_api.md): `PiPLSRegression` parameters, fitted attributes, and
   scikit-learn behavior;
5. [`path_analysis.md`](path_analysis.md): details of the cross-validated component and
   predictor-rank search;
6. [`cross_validation.md`](cross_validation.md): splitters, groups, repeated CV, OOF predictions,
   and validation reports;
7. [`api/index.md`](api/index.md): generated signatures, parameters, fitted attributes, shapes, and
   method contracts for the core public API.

The quickstart fixes both ranks and shows the estimator interface. The selection guides and
real-data examples show how to choose them.

## Fitted-model inspection

Model selection and fitted-model interpretation answer different questions. After choosing and
fitting a model, use:

- [`model_inspection.md`](model_inspection.md): latent scores, loadings, coefficients, prediction
  diagnostics, and the Pi-PLS-specific factorization;
- [`examples.md`](examples.md): numbered workflows from a fixed fit to complete analyses.

## Data and validation workflows

- [`preprocessing.md`](preprocessing.md): model-internal centering and scaling, fold-local fitting,
  and leakage boundaries;
- [`datasets.md`](datasets.md): optional dataset containers, deterministic synthetic generation,
  and the committed reference datasets;
- [`benchmarks.md`](benchmarks.md): focused synthetic benchmark questions and minimal result policy;
- [`reproducibility.md`](reproducibility.md): software, dataset, and validation reproducibility.

## Scientific and numerical background

- [`theory.md`](theory.md): implemented Pi-PLS construction, interpretation of the two ranks,
  fitted quantities, and numerical invariances.
