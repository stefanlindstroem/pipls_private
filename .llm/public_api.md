# Public API contract

## Current top-level API

```python
from pipls import (
    PiPLSDecomposition,
    PiPLSPathCV,
    PiPLSRegression,
    PiPLSValidationReport,
    StatisticalSupportWarning,
)
```

`PiPLSRegression` fits one explicit fixed pair `(n_components, predictor_rank)` and performs no
cross-validation or parameter selection. `PiPLSPathCV` is the standard package workflow for the
bounded triangular scan and conditional predictor-rank selection. Public scoring callables remain
available from `pipls.metrics`.

## Public names

| Public name | Mathematical notation |
|---|---|
| `n_components` | $h$ |
| `predictor_rank` | $r_\pi$ |
| `samples_per_predictor_rank` | $c$ in `PiPLSPathCV` |
| `predictor_rank_` | fitted explicit $r_\pi$ |
| `max_predictor_rank_` | centered algebraic limit on the fixed estimator; search ceiling on the path |

Do not expose constructor aliases named `h`, `r_pi`, or `c`.

## Fixed-estimator validation and warning contract

`PiPLSRegression` constructor parameters are `n_components`, `scale`, `copy`, `predictor_rank`,
`svd_solver`, and `random_state`.

- `n_components` and `predictor_rank` are positive Python or NumPy integers; booleans and
  integral-valued floats are invalid.
- `n_components <= predictor_rank`.
- `predictor_rank <= min(n_features, n_samples - 1)` after centering and must not exceed the
  verified numerical rank.
- `random_state` lies in $[0, 2^{32}-1]`. It may be `None` only with `svd_solver="full"`.
- `scale` and `copy` are Python or NumPy booleans.

A direct fixed fit emits `StatisticalSupportWarning` when $n/r_\pi<4$. This warning is diagnostic;
it does not choose or cap the rank. `PiPLSPathCV` suppresses only this expected warning inside its
controlled candidate and OOF fits.

## Model-internal standardization contract

Every `PiPLSRegression.fit` estimates `x_mean_` and `y_mean_` from the observations supplied to
that fit. `scale=True` estimates safe sample-standard-deviation vectors with `ddof=1` and
standardizes both blocks; `scale=False` still centers and stores unit scales. During path
selection, every candidate clone learns these statistics only from its training fold. The optional
final refit learns them again from all observations supplied to `PiPLSPathCV.fit`.

## Predictor SVD policy

`svd_solver` accepts `"full"`, `"randomized"`, or `"auto"`; the default is `"auto"`.
`random_state=0` makes randomized decomposition reproducible. Only the first SVD of the
centered/scaled predictor matrix may be randomized. The response-subspace and coupling SVDs remain
exact.

## Fitted fixed-estimator behavior

The estimator provides PLS-style `fit`, `predict(X, copy=True)`,
`transform(X, y=None, copy=True)`, tuple-valued `fit_transform(X, y)`, `inverse_transform`, and
scalar R2 `score`. It supports feature names and pandas output. Standard PLS-style fitted
attributes, coefficients, and scores remain available; Pi-PLS factorization output is canonicalized
in the frozen `decomposition_` object.

The fixed estimator does not expose `cv_results_`, `best_params_`, OOF predictions, validation
reports, or predictor-rank search diagnostics. `response_scale_for_scoring_` remains available for
package scorers. `predictor_rank_` is the requested fitted integer and `max_predictor_rank_` is
`min(n_features, n_samples - 1)` for the supplied centered training data.

## Path-analysis API

`PiPLSPathCV` owns all package model selection. It defaults to adaptive
`search_method="auto"`; explicit `"optimal"` exhaustively evaluates every admissible pair. The
surface satisfies

\[
1 \le h \le \min(q,r_{\pi,\max}), \qquad h \le r_\pi \le r_{\pi,\max}.
\]

The default ceiling uses total supplied $n$ for the support term with
`samples_per_predictor_rank=5`, while centered fold dimensions remain hard feasibility caps. The
class accepts a direct fixed `PiPLSRegression` or a pipeline ending in one, materializes one CV
split set, clones fixed candidates, and optionally refits the selected pair.

Public path attributes include standard search results, `best_pipls_`, conditional path and surface
diagnostics, immutable `validation_report_`, optional OOF outputs, and the canonical
`component_path_results_` table. The numeric predictor rank is present in every component-path row.

Group-aware splitters and keyword-only `groups` belong to `PiPLSPathCV.fit`, not to the fixed
estimator. `return_oof_predictions` and selection-conditioned reporting likewise belong only to the
path interface.

## E1 dataset and synthetic-data API

Dataset functionality is public from the dedicated `pipls.datasets` namespace:

```python
from pipls.datasets import (
    PiPLSDataset,
    PiPLSSyntheticTruth,
    make_pipls_regression,
    make_pipls_train_test,
)
```

`PiPLSDataset` is an optional immutable in-memory container, primarily useful for package-owned
synthetic data and structured experiments. Plain arrays and data frames passed directly to
`fit(X, Y)` remain the primary real-data interface. The container stores read-only `float64` `X` and
2D `Y`, unique feature/target/sample names, required provenance, recursively frozen metadata, and
optional synthetic truth. `data` and `target` are scikit-learn-style aliases. Required provenance
keys are `source`, `license`, `citation`, and `version`.

`make_pipls_regression` creates one side-effect-free dataset with local seeded random generation.
It supports shared, predictor-specific, and response-specific latent ranks; scalar or per-direction
strengths; normal or uniform source distributions; scalar or per-variable observed scales; and
scalar or separate predictor/response noise. `random_state=0` is the deterministic default and
must be an unsigned 32-bit integer. Each sample block must contain more rows than the larger
centered latent rank requested for `X` or `Y`.

`make_pipls_train_test` creates two datasets from one shared loading/strength/scale model and
independent train/test score and noise draws. It performs no fitted preprocessing and the training
block does not depend on the requested test size.

`PiPLSSyntheticTruth` exposes read-only latent scores, loading blocks, signal/noise matrices,
strengths, and scales. Loading blocks that are structurally absent are explicit zeros.

No metadata file, registry lookup, or package-owned loader is required for real-data fitting.
Users read and prepare `X` and `Y` with ordinary domain-appropriate code. Repository examples
must show these steps directly rather than hiding them behind convenience utilities.
