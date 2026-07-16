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

`PiPLSRegression.fit` accepts the same `groups=` metadata for its internal rank search. Because
`groups` is an explicit fit parameter, it also participates in scikit-learn metadata routing when
routing is enabled and requested.

Recommended splitters include:

- `RepeatedKFold` for repeated random folds;
- `PredefinedSplit` for an official train/test role;
- `GroupKFold` for grouped measurements;
- `TimeSeriesSplit` for ordered temporal data;
- `LeaveOneOut` only when the scientific protocol requires it.

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

`LeaveOneOut()` is not a special Pi-PLS mode. It produces the ordinary fold-safe bound using
`n_train_min = n - 1`. The default response-standardized MSE is valid for singleton folds because
response scales are estimated only from each training fold.

Ordinary foldwise R2 is rejected when any validation fold contains one sample. When OOF
predictions are requested, `pooled_oof_r2_` may be reported as **R2 from pooled LOO predictions**;
it is not mean foldwise R2.

For `PiPLSPathCV`, and for `PiPLSRegression` with `predictor_rank="auto"` or `"optimal"`, the same
CV results are used for selection and performance reporting. Therefore
`validation_report_.estimate_kind == "selection-conditioned"`. This estimate is useful for the
paper protocol but is not an unbiased post-selection estimate. Use nested CV or an external test
set for unbiased assessment.
