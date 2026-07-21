# Decision 0062: canonical Pulp workflow

## Status

Accepted and implemented.

## Context

The planned tutorial-first documentation needs one executable Pulp analysis shared by the numbered
example, generated tutorial figures, and later documentation excerpts. The existing Pulp script
performed data loading, path evaluation, fixed fitting, OOF prediction, inspection, and artifact
writing in one file. Reimplementing those numerical stages in a documentation renderer would create
drift.

Pi-PLS already learns predictor and response centering and optional scaling inside each fit. Adding
an external `StandardScaler` only to demonstrate a pipeline would either duplicate predictor
standardization or change the multivariate scaling model.

## Decision

`examples/_support/pulp_workflow.py` owns the canonical Pulp numerical workflow:

1. read the committed `X.csv` and `Y.csv` tables explicitly with pandas;
2. construct a cloneable scikit-learn `Pipeline` whose terminal `pipls` step is
   `PiPLSRegression`;
3. evaluate that complete pipeline with `PiPLSPathCV(refit=False)`;
4. read the conditional predictor rank from the row for the stated component count;
5. clone the pipeline, assign `pipls__n_components` and `pipls__predictor_rank`, and fit it on all
   observations;
6. clone the fixed selected pipeline inside five non-shuffled folds for aligned OOF predictions;
7. compute the immutable Pi-PLS factors, prediction diagnostics, and shared latent structure used by
   plots and tables.

The pipeline intentionally contains only the Pi-PLS estimator. It demonstrates the supported
pipeline composition boundary without introducing scientifically unjustified preprocessing.
Application-specific transformers may be inserted before the terminal step in other workflows.

The frozen `PulpWorkflowResult` groups the analysis products without moving them into the installed
package. `examples/10_pulp_real_data.py` consumes this result and remains responsible for canonical
CSV output, PDF rendering, printed summaries, and the visible three-component choice.

## Consequences

The Pulp example, future tutorial renderer, and future documentation excerpts can share one
numerical implementation. The package gains no dataset loader, tutorial object, or new public API.
Sugarcane and Tobacco remain independent complete examples. The next documentation increment may
generate tutorial figures from this workflow without duplicating model development logic.
