# Pi-PLS path selection

Use `PiPLSPathCV` to evaluate admissible `(n_components, predictor_rank)` pairs by cross-validation.
Methods that delegate to a selected estimator are available only when `refit=True`.

::: pipls.PiPLSPathCV
    options:
      members:
        - fit
        - predict
        - transform
        - fit_transform
        - inverse_transform
        - score
        - get_feature_names_out
