# Preprocessing semantics

`PiPLSRegression` always centers predictors and responses using statistics estimated from the data
used in the current fit.

With `scale=True`, each centered column is divided by its sample standard deviation using
`ddof=1`. A zero scale, and every scale estimated from a singleton training set, is replaced by 1.
With `scale=False`, centering remains active while `x_scale_` and `y_scale_` are vectors of ones.

Automatic predictor-rank selection fits these statistics independently inside every training
fold. Validation observations do not influence fold means or scales. After rank selection, the
estimator refits preprocessing and the selected fixed-rank Pi-PLS model on all data supplied to
`fit()`.

Response-standardized model-selection loss uses a separate
`response_scale_for_scoring_`. It is always the safe training-response sample standard deviation,
even when estimator preprocessing has `scale=False`.
