# Decision 0070: direct Pi-PLS and ordinary-PLS path comparison

## Status

Accepted and implemented.

## Context

Example 09 is the sole numbered workflow that fits ordinary PLS. Its purpose is
narrow: compare response-standardized CV-MSE against component count for Pi-PLS
and scikit-learn `PLSRegression` on the three reference datasets.

The comparison previously converted the Pi-PLS path to a pandas DataFrame,
returned the ordinary-PLS path as another DataFrame, wrote both tables as CSV
files, and passed their filenames to a separate plotting helper. This obscured
the two numerical paths and made generated files part of the analytical data
flow even though the final user product is the comparison figure.

The fold-local ordinary-PLS evaluator remains useful because it fits one maximum
NIPALS model per fold and obtains the nested path by truncating the fitted
rotations and response loadings. That calculation is nontrivial and should not
be duplicated in the numbered example.

## Decision

`examples/_support/pls_component_path.py` returns a frozen
`PLSComponentPath` with:

```text
n_components
cv_mse_mean
cv_mse_fold_sd
algorithm
n_splits
```

The three numerical fields are defensive, read-only, aligned NumPy arrays.
Component counts are positive, unique, and strictly ascending; CV-MSE means are
finite; fold standard deviations are finite and nonnegative. The algorithm and
split count are scalar metadata. Pickling reconstructs the object through its
validation path so array write protection is preserved.

`examples/09_pls_path_comparison.py` owns one visible workflow for each reference
dataset:

1. read `X.csv` and `Y.csv` directly with pandas;
2. fit `PiPLSSearchCV(refit=False)` and retain `component_path_`;
3. evaluate the matched ordinary-PLS path in memory;
4. verify that the two component-count arrays agree;
5. create one Matplotlib figure and axis;
6. plot both means with fold standard deviations;
7. annotate the conditionally selected Pi-PLS predictor ranks;
8. save and close the final comparison PDF.

The example writes only:

```text
pulp_component_path_comparison.pdf
sugarcane_component_path_comparison.pdf
tobacco_component_path_comparison.pdf
```

Delete `examples/_support/plot_component_path.py`. Do not write Pi-PLS or
ordinary-PLS comparison CSV files.

This decision supersedes the generated-table and CSV-rendering parts of
Decisions 0036 and 0047. Their durable ownership rule remains: example 09 is the
only numbered real-data Pi-PLS/ordinary-PLS component-path comparison, while
examples 10–12 remain Pi-PLS-only analyses.

## Consequences

- Every numbered real-data workflow now operates directly on in-memory results.
- The ordinary-PLS nested-path calculation remains isolated and testable without
  hiding the comparison or its figure composition.
- The comparison result object follows the repository's immutable-result style
  without adding a package-level public API.
- Generated CSV files are no longer analytical or plotting intermediates in any
  numbered example.
- The three comparison PDFs remain the final user-facing products.
- No Pi-PLS numerical behavior, estimator contract, or top-level export changes.
