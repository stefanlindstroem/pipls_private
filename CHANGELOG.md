# Changelog

## Unreleased

- Establish repository skeleton and LLM-assisted snapshot/patch workflow.
- Add the fixed-parameter Pi-PLS numerical core.
- Add the public fixed-rank `PiPLSRegression` estimator.
- Add the ceiling-based predictor-rank bound and `predictor_rank="max"` mode.
- Add private cross-validation selection primitives for reusable splits, response-standardized loss, and deterministic rank tie-breaking.
- Add automatic predictor-rank selection with fold-local preprocessing, diagnostics, public scoring utilities, and full-data refitting.
- Accept the roadmap split between exhaustive `"optimal"` rank search and adaptive `"auto"` search, with randomized SVD reserved for a separate solver policy.
