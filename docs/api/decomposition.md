# Pi-PLS decomposition

`PiPLSRegression.decomposition_` stores the interpretable Pi-PLS factorization and numerical-rank
diagnostics as immutable arrays. It is a fitted result object; users normally obtain it from a
fitted estimator rather than construct it directly.

::: pipls.PiPLSDecomposition
    options:
      show_signature: false
      members:
        - standardized_regression_map

The public fields use descriptive Python names:

| Field | Method notation | Meaning |
|---|---|---|
| `predictor_rotations` | $P$ | orthogonal predictor directions |
| `dilation` | $\operatorname{diag}(D)$ | nonnegative strength of each paired mode |
| `response_rotations` | $Q$ | orthogonal response directions |
| `standardized_regression_map` | $PDQ^{\mathsf T}$ | regression map in centered/scaled coordinates |

The private construction also uses the truncated predictor basis $\Pi$, response basis $C$, and
least-squares map $W$. Those intermediate matrices are not public results because normal fitting,
inspection, prediction, and plotting workflows do not consume them.

`PiPLSRegression` transforms the centered/scaled regression map back to the original predictor and
response units when it constructs `coef_`, `intercept_`, and `predict()` output.
