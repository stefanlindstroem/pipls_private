# Decision 0090: consolidate the programming reference

## Status

Accepted.

## Context

The served reference had grown to thirteen flat navigation entries. Four generated API pages
contained only one small result or utility group, while path-selection behavior was split between
two adjacent guides. `model_inspection.md` repeated elementary Matplotlib recipes already visible in
the tutorials and maintained examples. Some served prose also described removed pre-release
surfaces rather than the supported positive contract.

The split produced unnecessary navigation choices and duplicated scoring, preprocessing, plotting,
and validation explanations. It also left one inaccurate statement that `cv_results_` retained
resolved grids and adaptive-search history.

## Decision

Consolidate the reference without changing the public API or numerical behavior.

1. Keep eight reference entries: API overview, fixed regression, path selection, path-selection
   details, troubleshooting, model inspection, inspection API, and datasets/synthetic data.
2. Document `PiPLSDecomposition` and `StatisticalSupportWarning` with fixed regression.
3. Document `PiPLSValidationReport` and the public scoring functions with path selection.
4. Merge advanced path-search and cross-validation behavior into one path-selection details page.
5. Retain every stable inspection anchor and scientific interpretation boundary while replacing
   elementary chart recipes with one compact quantity catalogue and concise interpretation notes.
6. Describe only the supported positive contract in served pages. Historical removals remain in
   decisions and the guide layer.
7. State the actual `cv_results_` contents: evaluated candidate parameters, split and summary
   scores, response-standardized MSE diagnostics, score ranks, and timing summaries.

Generated mkdocstrings directives remain present for every public object. The removed Markdown
files are navigation wrappers, not removed APIs.

## Consequences

Programming users face fewer navigation choices and one canonical source for path selection and
validation behavior. The model-inspection reference remains scientifically complete and keeps all
tutorial link targets, but no longer duplicates routine Matplotlib syntax. Documentation tests
protect public-object coverage, merged navigation, stable links and anchors, and the absence of the
retired wrapper pages.

A later owner-directed audit inserted a six-patch hardening series before any release work;
Decision 0091 begins that series.
