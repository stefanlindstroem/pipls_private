# Cross-validation protocols and OOF reporting

The tutorial uses ordinary five-fold validation. `PiPLSPathCV` also accepts scikit-learn splitters
for grouped, repeated, predefined, temporal, or leave-one-out protocols:

```python
from sklearn.model_selection import GroupKFold

from pipls import PiPLSPathCV

search = PiPLSPathCV(cv=GroupKFold(n_splits=5), refit=False)
search.fit(X, Y, groups=sample_groups)
```

`groups` is an explicit `PiPLSPathCV.fit()` parameter and participates in scikit-learn metadata
routing when routing is enabled and requested. Split metadata belongs to the path selector;
`PiPLSRegression.fit(X, Y)` fits one explicit pair and accepts none.

Suitable splitters include `RepeatedKFold`, `PredefinedSplit`, `GroupKFold`, `TimeSeriesSplit`, and
`LeaveOneOut` when their scientific assumptions match the data.

## Scoring

The default callable is `pipls.metrics.neg_response_standardized_mean_squared_error`. Candidate
selection always maximizes the configured mean test score. With the default scorer this is
minimization of mean response-standardized CV-MSE; with another scorer, the CV-MSE columns are
retained as descriptive diagnostics.

Ordinary scikit-learn scorer names, other callables, and `scoring=None` are accepted. The package
does not define a private scorer-name registry.

## Ordered out-of-fold predictions

Set `return_oof_predictions=True` to fit the selected fixed parameterization once per training fold
after selection. The immutable `validation_report_` then owns the row-ordered OOF results.

- `validation_report_.oof_predictions` preserves input row order.
- Rows validated once have count 1.
- Repeated validation predictions are averaged and recorded in
  `validation_report_.oof_prediction_counts`.
- Rows never used for validation have count 0 and a NaN OOF prediction.
- `validation_report_.n_components` and `predictor_rank` identify the fitted pair.
- `validation_report_.pooled_oof_r2` is calculated only on rows with OOF coverage.

The same report records the split count, mean score, CV-MSE, OOF coverage, pooled OOF $R^2$, and
whether the splitter is structurally leave-one-out.

These results are selection-conditioned because the same path search selected the parameters. They
are not an unbiased post-selection estimate. Use nested cross-validation or an external test set
when that stronger claim is required.

## Leave-one-out interpretation

`LeaveOneOut()` is not a special Pi-PLS mode. The support term uses the full $n$, while centered-fold
feasibility is capped by $n-2$. The default standardized MSE remains defined for singleton
validation folds because response scales are estimated from each training fold.

Mean foldwise $R^2$ is rejected when validation folds contain one sample. When OOF predictions are
requested, `validation_report_.pooled_oof_r2` may report $R^2$ from pooled LOO predictions; it is
not mean foldwise $R^2$.

## Fold variation

`component_path_.cv_mse_fold_sd` is the population standard deviation of the realized fold-specific
MSE values. It describes fold variation, not a confidence interval. Formal uncertainty statements
require a separately designed repeated or nested resampling procedure.
