# Pi-PLS documentation

This documentation describes the installable `pipls` package, its implemented theory, and its
supported user workflows.

## Start here

1. [`quickstart.md`](quickstart.md): literal NumPy matrices, one fixed fit, predictions, and one
   decomposition plot;
2. [`examples.md`](examples.md): synthetic, comparison, and complete real-data workflows;
3. [`estimator_api.md`](estimator_api.md): `PiPLSRegression` parameters, fitted attributes, and
   scikit-learn behavior;
4. [`parameter_selection.md`](parameter_selection.md): component and predictor-rank controls,
   scoring, and selection policy;
5. [`path_analysis.md`](path_analysis.md): joint `n_components` and `predictor_rank` search;
6. [`cross_validation.md`](cross_validation.md): splitters, groups, repeated CV, OOF predictions,
   and validation reports.

The quickstart is the ordinary fixed-model entry point. It does not require pandas, a repository
dataset, or cross-validation. The complete real-data examples are later-stage reference workflows.

## Fitted-model inspection

Component-path tables diagnose model selection. Fitted-model decomposition, score, loading, and
coefficient views are a separate interpretation stage, while observed-versus-predicted and residual
diagnostics require explicit prediction provenance.

- [`model_inspection.md`](model_inspection.md): Pi-PLS display factors, prediction diagnostics,
  Pi-PLS-specific factorization inspection, shared PLS-family analysis, and optional Matplotlib
  figures;
- [`examples.md`](examples.md): numbered examples and the distinction between introductory scripts,
  explicit comparison, and complete analyses.

## Data and validation workflows

- [`preprocessing.md`](preprocessing.md): model-internal centering and scaling, fold-local fitting,
  and leakage boundaries;
- [`datasets.md`](datasets.md): optional dataset containers, deterministic synthetic generation,
  and the committed reference datasets;
- [`benchmarks.md`](benchmarks.md): focused synthetic benchmark questions and minimal result policy;
- [`reproducibility.md`](reproducibility.md): software, dataset, and validation reproducibility.

## Scientific and numerical background

- [`theory.md`](theory.md): implemented Pi-PLS construction, rank interpretation, fitted quantities,
  and numerical invariances.

## Maintainer records

Accepted engineering decisions are indexed under [`decisions/index.md`](decisions/index.md). They
record why contracts were adopted, but they are not prerequisites for using the package.
