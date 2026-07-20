# Metrics

The public scoring functions standardize each response residual by the corresponding sample
standard deviation learned from the estimator's training responses. The positive function reports
an error; the negative function follows the scikit-learn convention that larger scorer values are
better.

The path estimator uses the negative form by default, so maximizing its mean validation score is
equivalent to minimizing response-standardized MSE.

::: pipls.metrics.response_standardized_mean_squared_error
    options:
      members: false

::: pipls.metrics.neg_response_standardized_mean_squared_error
    options:
      members: false
