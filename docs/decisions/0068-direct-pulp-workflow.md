# Decision 0068: direct Pulp workflow

## Status

Accepted and implemented.

## Context

The Pulp example previously delegated most of its analysis to
`examples/_support/pulp_workflow.py`. The wrapper fitted the selection path,
selected a parameter pair, fitted the fixed model, calculated out-of-fold
predictions, and assembled inspection results.

Although convenient for repository rendering, that abstraction hid the
workflow that a programming user needs to understand and modify. It also
continued the transitional pattern of converting analytical results to tables
and using generated CSV files between calculation and plotting.

The Pulp dataset additionally requires careful interpretation of predictor
rank. For the selected component count $h=3$, the best evaluated predictor
rank is $r_\pi=10$. This rank is at the upper search boundary and must not be
confused with the number of paired Pi-PLS components.

## Decision

`examples/10_pulp_real_data.py` owns one visible linear workflow:

1. read `X.csv` and `Y.csv` directly with pandas;
2. fit `PiPLSPathCV(refit=False)`;
3. inspect and plot `component_path_` directly;
4. choose $h=3$ with `for_n_components()`;
5. inspect the predictor-rank profile conditional on $h=3$;
6. retrieve the selected predictor rank $r_\pi=10$;
7. fit one fixed `PiPLSRegression` model;
8. calculate five-fold out-of-fold predictions with
   `cross_val_predict()`;
9. calculate the immutable inspection results in memory;
10. create and save each final figure explicitly.

The example writes only final PDF figures:

```text
component_path.pdf
predictor_rank_profile.pdf
pipls_factors.pdf
latent_structure.pdf
coefficients.pdf
prediction_diagnostics.pdf
```

The shared `PulpWorkflowResult` and `run_pulp_workflow()` abstraction are
removed. The tutorial extracts checked source snippets from the numbered
example. Its figure renderer performs the required calculations directly from
the public API and does not depend on generated analytical CSV files.

The predictor-rank profile states that ranks 3 through 10 were evaluated for
$h=3$, that rank 10 has the lowest evaluated mean CV-MSE, and that rank 10 is
the upper evaluated boundary. The factor figures continue to show three paired
components, not a ten-component model.

## Consequences

- The complete Pulp analysis is visible in the numbered example.
- Users can modify path selection, fixed fitting, prediction, inspection, and
  plotting without following a repository-specific workflow wrapper.
- Predictor-rank selection is presented with its evaluated range and fold
  variability rather than as an unqualified optimum.
- The tutorial and its generated figures depend on in-memory public results.
- Tobacco retains the remaining post-analysis table machinery until Patch 20d.
- No package numerical behavior or public estimator contract changes.
