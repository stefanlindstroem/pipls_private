# Decision 0034: two-stage component-path workflow

## Status

Accepted and implemented. The real-data benchmark-script and full example-execution-test portions are superseded by Decision 0037. The concise public result representation is superseded by Decision 0066.

## Context

The first Pulp and Sugarcane smoke checks reduced a two-parameter path search to one globally
selected row. Their examples printed `best_params_` and a selection-conditioned OOF diagnostic.
That presentation could be read as an instruction to let cross-validation make the final scientific
choice of `n_components` automatically.

The intended workflow is different. Predictor rank is selected or supplied conditionally for each
candidate component count. The resulting CV-MSE path is inspected by the user, who chooses a
component count using predictive loss, uncertainty, parsimony, and interpretation. A separate fixed
Pi-PLS model is then fitted on all observations with both ranks stated explicitly.

## Decision

`PiPLSPathCV` exposes one conditionally selected result per requested `n_components` value.
Decision 0066 replaces the original dictionary representation with the immutable
`component_path_` object. Its aligned fields are:

```text
n_components
predictor_rank
predictor_rank_policy
mean_test_score
cv_mse_mean
cv_mse_fold_sd
n_splits
```

`predictor_rank` is always numeric. `predictor_rank_policy` uses `optimized`, `fixed`, or
`maximum`. `predictor_rank_values=None` searches the ordinary conditional path, a one-element
sequence fixes a rank, a longer sequence searches within that set, and
`predictor_rank_values="max"` uses the rule-derived maximum directly for every component count.

The uncertainty column is the population standard deviation of the fold MSE values already stored
in `cv_results_`. It describes fold-to-fold variation and is not a confidence interval or an
independent standard error because CV training sets overlap.

Real-data path examples use `refit=False`, write the component path to CSV, and generate a PDF by
reading that CSV. The CSV is canonical; the plot does not receive a fitted search object. The user
then chooses `n_components` through a visible constant, reads the matching numeric predictor rank
from the CSV, and fits a separate `PiPLSRegression` with both values fixed. Fold SD remains
descriptive rather than becoming a formal selection threshold.

The public real-data examples write the six-column Pi-PLS component-path CSV. Earlier duplicate
smoke benchmark copies were removed by Decision 0037.

## Consequences

- `best_params_` remains available for scikit-learn compatibility and programmatic convenience, but
  examples do not present it as the required scientific decision.
- Every component-path artifact records the actual predictor rank even when rank is fixed or set to
  the rule-derived maximum.
- CSV output remains the source artifact; example PDFs are derived views and are ignored by Git.
- A final model reproduces one evaluated path row because both `n_components` and `predictor_rank`
  are fixed explicitly.
- Component count remains an explicit path-based user choice.
- Formal uncertainty estimates requiring repeated or nested resampling require a separate design.
