# Decision 0044: establish a minimal onboarding path and separate example support code

## Status

Accepted and implemented.

## Context

The estimator, selection, inspection, and complete real-data workflows are implemented, but the
visible examples begin with advanced cross-validation and synthetic-data generation. The README has
a short estimator fragment, yet it depends on undefined training and test variables. A user browsing
`examples/` may therefore encounter the complete report infrastructure before seeing the ordinary
scikit-learn pattern of constructing arrays, fitting one model, and predicting.

The complete real-data workflows also keep four substantial helper modules directly beside the
numbered scripts. Those helpers are justified for reproducible CSV and report generation, but their
placement makes them appear to be primary user entry points.

## Decision

Add `examples/01_minimal_fit_and_plot.py` and a matching `docs/quickstart.md`. The example uses
literal NumPy matrices, explicit predictor and response names, one fixed `PiPLSRegression` fit, one
`predict()` call, and one decomposition plot. It performs no cross-validation, path search, pandas
I/O, ordinary PLS comparison, or artifact-table construction.

Keep fitting, numerical inspection, and rendering as separate calls. Do not add an estimator-owned
`plot()` method or a combined fit-and-plot convenience API. Matplotlib remains optional.

Move the complete-workflow helpers under `examples/_support/`. The numbered scripts continue to
import and use them, but the underscore-prefixed package identifies them as implementation support
rather than the recommended starting point. The helpers remain outside `src/pipls/` because they
own example-specific OOF loops, pandas tables, CSV files, physical-axis metadata, pagination, and
multipage report composition.

Organize user navigation into four levels:

1. direct fixed-model quickstart;
2. model selection and cross-validation;
3. fitted-model inspection;
4. complete Pulp, Sugarcane, and Tobacco analyses.

Retain the single `make examples` target. Do not add a separate quick-example target.

## Consequences

The repository now has a runnable first example that resembles ordinary scikit-learn use and makes
clear that complete report generation is optional. The real-data workflows and their artifact
contracts remain unchanged. Estimator defaults, single-response behavior, selection rules, and
numerical results are not changed by this decision; any change to those public contracts requires a
separate decision.
