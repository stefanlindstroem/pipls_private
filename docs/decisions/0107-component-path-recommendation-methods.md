# Decision 0107: component-path recommendation methods

## Status

Accepted.

## Context

`PiPLSComponentPath` stores one aligned, conditionally selected result for every evaluated component
count. It already provides the component count, associated predictor rank and policy, mean configured
score, mean response-standardized CV-MSE, fold dispersion, split count, and derived fold-based
standard error for each row.

Programming users may want concise references for the minimum-CV-MSE row and the conventional
one-standard-error recommendation without reconstructing those rules manually. These references
must remain result inspection only: they must not mutate the path search, replace `best_*`, fit or
refit an estimator, or become the promoted tutorial workflow.

## Decision

`PiPLSComponentPath` exposes two non-mutating methods:

- `minimum_cv_mse_result()` returns the stored row with the smallest `cv_mse_mean`;
- `one_standard_error_result()` returns the stored row with the smallest evaluated component count
  whose `cv_mse_mean` does not exceed the minimum mean plus the fold-based standard error from that
  minimum row.

Both methods return `PiPLSComponentResult`, so the recommendation includes the aligned component
count, conditionally selected predictor rank and policy, score, CV-MSE summaries, and split count.
The predictor rank is not selected again. It is the rank already retained in the component-path row
under the configured scorer. With a nondefault scorer, that rank need not minimize CV-MSE within its
component-count profile.

The methods operate directly on the immutable stored arrays. `np.argmin()` identifies the first
minimum row; because `n_components` is strictly ascending, an exact tie returns the smallest tied
component count. The one-standard-error threshold and eligibility comparison use the stored
floating-point values exactly. No numerical tolerance, compatibility rule, or additional stored
recommendation state is introduced.

`minimum_cv_mse_result()` remains defined for a valid one-split path. The one-standard-error method
requires the minimum row to have at least two validation splits, through the existing
`cv_mse_standard_error` contract, and raises if the derived threshold is nonfinite.

The methods are listed in the generated path API reference. Numbered examples, tutorial renderers,
tutorial prose, the README, and the documentation home continue to use explicit component choices
and do not promote or invoke these methods.

## Consequences

Programming users can retrieve complete aligned recommendations without manual indexing or
repeating the one-standard-error calculation. The methods add no fitted state, constructor fields,
serialization data, estimator delegation, or automatic model-selection behavior. A following
reference-documentation increment may explain the methods in `docs/path_analysis.md` without adding
them to maintained examples or tutorials.
