# Preprocessing semantics

`PiPLSRegression` always centers predictors and responses using statistics estimated from the data
used in the current fit.

With `scale=True`, each centered column is divided by its sample standard deviation using
`ddof=1`. A zero scale for a constant column, and every scale estimated from a singleton training
set, is replaced by 1. Ordinary NumPy means and standard deviations are retained for ordinary data;
a range-safe fallback is used only when finite values overflow those calculations or when a
nonconstant column would otherwise underflow to a zero scale. With `scale=False`, centering remains
active while `x_scale_` and `y_scale_` are vectors of ones.

During `PiPLSPathCV` selection, every fixed candidate learns these statistics independently inside
each training fold. Validation observations do not influence fold means or scales. With
`refit=True`, the path meta-estimator learns preprocessing again while fitting the selected fixed
pair on all data supplied to `PiPLSPathCV.fit()`.

Response-standardized model-selection loss uses a separate
`response_scale_for_scoring_`. It is always the safe training-response sample standard deviation,
even when estimator preprocessing has `scale=False`.

## Fit-state and input-storage safety

`PiPLSRegression.fit()` and `PiPLSPathCV.fit()` are transactional. A failed fit leaves the object
unfitted and removes any earlier fitted state. With `copy=False`, writable independent arrays may be
modified in place, while read-only arrays and overlapping predictor/response views are copied as
needed. Successful public fitted values and outputs must be finite.
