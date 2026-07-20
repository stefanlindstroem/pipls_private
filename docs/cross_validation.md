# Cross-validation protocols and OOF reporting

Pi-PLS accepts ordinary scikit-learn splitters. Split policy remains outside the numerical core:

```python
from sklearn.model_selection import GroupKFold, LeaveOneOut, TimeSeriesSplit
from pipls import PiPLSPathCV

search = PiPLSPathCV(
    cv=LeaveOneOut(),
    return_oof_predictions=True,
)
search.fit(X, Y)
```

Use `groups=` with group-aware splitters:

```python
search = PiPLSPathCV(cv=GroupKFold(n_splits=5))
search.fit(X, Y, groups=sample_groups)
```

Cross-validation splitters, `groups`, scoring, and OOF reporting belong to `PiPLSPathCV`.
`PiPLSRegression.fit(X, Y)` fits one explicit fixed pair and accepts no split metadata.
Because `groups` is an explicit `PiPLSPathCV.fit` parameter, it participates in scikit-learn
metadata routing when routing is enabled and requested.

Recommended splitters include:

- `RepeatedKFold` for repeated random folds;
- `PredefinedSplit` for an official train/test role;
- `GroupKFold` for grouped measurements;
- `TimeSeriesSplit` for ordered temporal data;
- `LeaveOneOut` only when the scientific protocol requires it.


## Scoring

The default path scorer is the public callable
`pipls.metrics.neg_response_standardized_mean_squared_error`. It follows the ordinary
`(estimator, X, y)` scorer protocol and can be imported wherever a scorer callable is accepted:

```python
from pipls.metrics import neg_response_standardized_mean_squared_error

search = PiPLSPathCV(scoring=neg_response_standardized_mean_squared_error)
```

Ordinary scikit-learn scorer names, other callables, and `scoring=None` remain supported. The
package does not define a package-local scorer string that could be mistaken for a scorer
registered globally by scikit-learn. Predictor-rank and overall selections among evaluated
candidates always maximize the configured mean test score. `component_path_results_` still
reports response-standardized MSE, so with a nondefault scorer that MSE is diagnostic rather than
necessarily minimized.

## Ordered out-of-fold predictions

Set `return_oof_predictions=True` to fit the selected fixed parameterization once per training
fold after selection. The resulting `oof_predictions_` preserves input row order.

- A row validated once has count 1.
- Repeated validation predictions are averaged and recorded in `oof_prediction_counts_`.
- A row never used for validation has count 0 and a NaN OOF prediction. This is expected for some
  `PredefinedSplit` and `TimeSeriesSplit` configurations.
- `oof_params_` identifies the parameterization that generated the predictions.
- `pooled_oof_r2_` is calculated on rows with at least one OOF prediction and is only a pooled
  diagnostic.

The immutable `validation_report_` is a `PiPLSValidationReport`. It records the selected component
count and predictor rank, split count, mean selection score, response-standardized MSE, OOF
coverage, pooled OOF R2, and whether the splitter is structurally leave-one-out.

## Leave-one-out interpretation

`LeaveOneOut()` is not a special Pi-PLS mode. The samples-per-rank term uses the full $n$, while
the centered-fold feasibility cap is `n_train_min - 1 = n - 2`. The default
response-standardized MSE is valid for singleton validation folds because response scales are
estimated only from each training fold.

Ordinary foldwise R2 is rejected when any validation fold contains one sample. When OOF
predictions are requested, `pooled_oof_r2_` may be reported as **R2 from pooled LOO predictions**;
it is not mean foldwise R2.

For `PiPLSPathCV`, the same
CV results are used for selection and performance reporting. Therefore
`validation_report_.estimate_kind == "selection-conditioned"`. This is not an unbiased
post-selection estimate. Use nested CV or an external test set for unbiased assessment.

## Fold variation in component-path tables

`PiPLSPathCV.component_path_results_` reports
`response_standardized_cv_mse_fold_sd` for each component count. This is the population standard
deviation of the fold-specific MSE values already stored in `cv_results_`. It is useful for seeing
how much the validation loss varies across the chosen folds.

It is not a confidence interval. Formal uncertainty statements require a separately designed
repeated or nested resampling procedure.
