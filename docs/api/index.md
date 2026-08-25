# API reference

Reference is for exact object, result, and behavior lookup. The
[tutorials](../tutorials/quick_start.md) own worked modeling workflows; generated API sections on
these pages come directly from the public NumPy-style docstrings.

## Estimators

- [`PiPLSRegression`](regression.md) fits one explicit $(h,r_\pi)$ pair and documents its fitted
  decomposition and direct-fit support warning.
- [`PiPLSSearchCV`](path.md) evaluates admissible component/rank candidates and exposes explicit
  selection, OOF-reporting, and refitting operations.

For Π-PLS, `n_components` is the number of paired latent modes $h$, while `predictor_rank` is the
retained predictor-subspace dimension $r_\pi$. Their separate roles are defined under
[Interpretation of the two rank controls](../theory.md#interpretation-of-the-ranks).

## Results and supporting APIs

- [Selection and validation](../selection_validation.md) defines component paths, predictor-rank
  evidence, selections, OOF reports, scoring, CV metadata, and exact selection rules.
- [Model inspection](../model_inspection.md) defines fitted latent quantities, Π-PLS display
  factors, biplot coordinates, and prediction and observation diagnostics.
- [Datasets and generators](datasets.md) documents packaged datasets, immutable data/truth records,
  and synthetic generators.
- [Troubleshooting](../troubleshooting.md) covers warnings, validation failures, lifecycle errors,
  numerical failures, copying behavior, and unexpectedly expensive searches.

Mathematical sections use $\mathbf{X}$ and $\mathbf{Y}$. Python follows the scikit-learn convention
`fit(X, y)`, where `y` may be either one-dimensional or a two-dimensional multivariate response
matrix.

The runtime API returns numerical objects and contains no plotting submodule. Maintained examples
render inspection results with ordinary Matplotlib.
