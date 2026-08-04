# Decision 0036: real-data component-path comparison with standard PLS

## Status

Accepted, with comparison-example ownership superseded by Decision 0047. The numerical comparison contract remains implemented. The real-data benchmark-script and full example-execution-test portions are superseded by Decision 0037.

## Context

The Pulp, Sugarcane, and Tobacco examples already write a canonical Pi-PLS component-path CSV,
generate a PDF from that table, and fit a separate fixed Pi-PLS model after a visible user choice.
Showing only Pi-PLS makes it difficult for a programming user to judge how the two-rank method
relates to the nearest standard PLS regression workflow.

The comparison must not replace the two-stage Pi-PLS workflow, hide the predictor rank, or present
a selected component count as an automatic scientific conclusion. The CSV files remain canonical,
and uncertainty remains descriptive fold-to-fold variation rather than a confidence interval.

The Tobacco path also became unnecessarily slow when it was used as the real-data randomized-SVD
demonstration. Randomized predictor SVD is already covered by the dedicated solver-consistency
benchmark, so the Tobacco example does not need to repeat that role.

## Decision

`examples/09_pls_path_comparison.py` writes two separate path tables for each reference dataset:

- `<dataset>_component_path.csv` for Pi-PLS, retaining the established six-column schema with the
  numeric predictor rank on every row;
- `<dataset>_pls_component_path.csv` for scikit-learn `PLSRegression`, with columns
  `n_components`, `algorithm`, mean response-standardized CV-MSE, fold SD, and split count.

The standard PLS implementation is labeled `NIPALS`, matching scikit-learn `PLSRegression`. Both
methods use the same ordered five-fold partition, fold-local centering and scaling, component
counts, response-standardized loss, and population SD across fold losses.

The PLS path helper fits the maximum requested NIPALS model once per fold and evaluates every
nested component count by truncating the public fitted weights and loadings. NIPALS extracts
components sequentially, so this is numerically equivalent to separately fitting each earlier
component count while avoiding repeated high-dimensional decompositions.

`examples/_support/plot_component_path.py` reads both CSV files. Its comparison function plots
`$\Pi$-PLS` and `PLS (NIPALS)` in one PDF, shows fold-SD error bars for both paths, and annotates
the Pi-PLS predictor rank beside every Pi-PLS point. The lower y-limit is zero and the upper limit
is data-driven but never below one.

Examples 10–12 retain a separate Pi-PLS-only path and final-model stage. The user chooses
`n_components`, reads the corresponding predictor rank from the Pi-PLS CSV, and fits a separate fixed
`PiPLSRegression` on all observations.

The Tobacco example uses explicit full predictor SVD with adaptive rank scanning.
Randomized-SVD behavior remains protected by the focused solver-consistency benchmark. The former
Tobacco smoke benchmark copy was removed by Decision 0037.

## Consequences

- The examples provide an immediate standard-PLS reference without turning the repository into a
  publication comparison suite.
- Pi-PLS and PLS results remain separate, machine-readable CSV artifacts before plotting.
- Predictor rank remains explicit for every Pi-PLS path row; standard PLS has no corresponding
  predictor-rank parameter.
- The overlaid PDF supports interpretation but is generated only from the canonical CSV files.
- No superiority claim, formal confidence interval, nested CV, preprocessing comparison, or final
  PLS model recommendation is added.
- The retired Tobacco randomized-path precursor is replaced for solver choice and bounded
  randomized-SVD demonstration.
