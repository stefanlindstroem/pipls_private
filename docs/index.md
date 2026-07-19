# Pi-PLS documentation

This documentation describes the installable `pipls` package and its supported user workflows.
The tracked `.llm/` layer contains maintainer contracts and roadmap state; it is not the primary
user guide.

## Start here

1. [`quickstart.md`](quickstart.md): literal NumPy matrices, one fixed fit, predictions, and one
   decomposition plot;
2. [`estimator_api.md`](estimator_api.md): `PiPLSRegression` parameters, fitted attributes, and
   scikit-learn behavior;
3. [`parameter_selection.md`](parameter_selection.md): component and predictor-rank controls,
   scoring, and selection policy;
4. [`path_analysis.md`](path_analysis.md): joint `n_components` and `predictor_rank` search;
5. [`cross_validation.md`](cross_validation.md): splitters, groups, repeated CV, OOF predictions,
   and validation reports.

The quickstart is the ordinary fixed-model entry point. It does not require pandas, a repository
dataset, or cross-validation. The complete real-data examples are later-stage reference workflows.

## Fitted-model inspection

Component-path tables diagnose model selection. Fitted-model decomposition, score, loading, and
coefficient views are a separate interpretation stage, while observed-versus-predicted and residual
diagnostics require explicit prediction provenance.

- [`model_inspection.md`](model_inspection.md): Pi-PLS display factors, prediction diagnostics,
  ordinary PLS analysis, and optional Matplotlib figures;
- [`../examples/README.md`](../examples/README.md): numbered examples, complete reference analyses,
  and the role of `examples/_support/`.

## Data and validation workflows

- [`preprocessing.md`](preprocessing.md): model-internal centering and scaling, fold-local fitting,
  and leakage boundaries;
- [`datasets.md`](datasets.md): optional dataset containers and deterministic synthetic generation;
- [`benchmarks.md`](benchmarks.md): focused synthetic benchmark questions and minimal result policy;
- [`../datasets/README.md`](../datasets/README.md): committed reference-dataset layout and provenance.

## Scientific and numerical background

- [`theory.md`](theory.md): user-facing theory overview;
- [`reproducibility.md`](reproducibility.md): software, dataset, and lightweight validation
  reproducibility.

## Development records

Accepted design decisions are stored under [`decisions/`](decisions/). They explain why public
contracts exist, but historical publication context does not define the package roadmap.
