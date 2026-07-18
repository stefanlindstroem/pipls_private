# Pi-PLS documentation

This documentation describes the installable `pipls` package and its supported user workflows.
The tracked `.llm/` layer contains maintainer contracts and roadmap state; it is not the primary
user guide.

## Start here

- [`../README.md`](../README.md): installation, first model, datasets, and repository map;
- [`estimator_api.md`](estimator_api.md): `PiPLSRegression` interface and fitted attributes;
- [`parameter_selection.md`](parameter_selection.md): component and predictor-rank controls,
  scoring, and selection policy;
- [`path_analysis.md`](path_analysis.md): joint `n_components` and `predictor_rank` search;
- [`cross_validation.md`](cross_validation.md): splitters, groups, repeated CV, OOF predictions, and
  validation reports.

## Data and model fitting

- [`preprocessing.md`](preprocessing.md): model-internal centering and scaling, fold-local fitting,
  and leakage boundaries;
- [`datasets.md`](datasets.md): optional dataset containers and deterministic synthetic generation;
- [`benchmarks.md`](benchmarks.md): focused synthetic benchmark questions and minimal result policy;
- [`../examples/README.md`](../examples/README.md): executable synthetic and real-data examples;
- [`../datasets/README.md`](../datasets/README.md): committed reference-dataset layout and provenance.

## Scientific and numerical background

- [`theory.md`](theory.md): user-facing theory overview;
- [`reproducibility.md`](reproducibility.md): software, dataset, and lightweight validation
  reproducibility.

## Development records

Accepted design decisions are stored under [`decisions/`](decisions/). They explain why public
contracts exist, but historical publication context does not define the package roadmap.
