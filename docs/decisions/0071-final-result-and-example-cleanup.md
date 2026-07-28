# Decision 0071: final result and example cleanup

## Status

Accepted and implemented.

## Context

Patches 20a–20e replaced the former application-layer table exchange with immutable numerical
results and direct example-owned plotting. `PiPLSSearchCV.component_path_` is now the concise
component-count view, while `cv_results_` remains the complete candidate-level dictionary of
aligned arrays.

Two fitted matrix aliases, `response_standardized_mse_path_` and `score_path_`, still duplicated
columns already present in `cv_results_`. No maintained package, documentation, example, tutorial,
or benchmark workflow used them. Keeping the aliases would require a second representation of the
same adaptive-search surface, including an implicit dense layout and NaN placeholders for
unevaluated cells.

The repository also needed one durable policy test that protects the completed direct-workflow
boundary. Earlier focused tests covered individual examples, but they did not collectively prevent
a later numbered example from reintroducing analytical CSV output, file-based plotting, or removed
helper APIs.

## Decision

Remove the fitted attributes:

```text
response_standardized_mse_path_
score_path_
```

Remove the private dense-surface construction helper used only by those attributes.
`cv_results_` is the sole detailed candidate-level search surface. Advanced users may reshape its
aligned `n_components`, `predictor_rank`, score, and response-standardized-MSE columns when a dense
matrix is useful. `component_path_` remains the immutable concise view with one conditionally
selected predictor rank per component count.

Adopt the following repository policy:

- numbered examples operate on in-memory arrays or immutable result objects;
- numbered examples do not call `DataFrame.to_csv()` or construct generated analytical tables;
- `pd.read_csv()` in numbered examples is limited to committed dataset inputs named `X.csv` and
  `Y.csv`;
- numbered real-data examples create their own Matplotlib figures and use `component_path_`;
- Pulp, Sugarcane, and Tobacco calculate OOF predictions and inspection results directly;
- example support code does not read serialized analytical results or own plotting output;
- tutorial figures are generated from the same direct public-API calculation, not from generated
  analysis files;
- `examples/results/` contains tracked directory placeholders and ignored final PDFs, not tracked
  CSV products.

Focused benchmark CSV files remain valid final machine-readable benchmark products. Committed
`datasets/*/X.csv` and `Y.csv` files remain input assets. User applications remain free to export
results independently.

Add structural tests for the policy above and for the absence of the removed result attributes and
obsolete helper functions. Historical decision records remain unchanged as maintainer history;
this decision supersedes their generated-analysis-table requirements where they conflict with the
current direct-workflow contract.

## Consequences

- `PiPLSSearchCV` has one concise result surface and one detailed result surface rather than three
  synchronized representations.
- Adaptive-search omissions remain explicit candidate absence in `cv_results_` instead of dense
  NaN cells.
- Numbered examples cannot silently regress to generated CSV intermediates or file-based plotting.
- The public documentation consistently directs ordinary users to `component_path_` and advanced
  users to `cv_results_`.
- The removal is intentionally made before the first tagged release, so no released compatibility
  contract is broken.
- Phase F4 is complete; first-release preparation is the next repository increment.
