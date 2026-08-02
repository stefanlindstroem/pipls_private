# Decision 0128: same-file rendering functions for complete examples

## Status

Accepted and implemented.

## Context

The Pulp example is the canonical sequential tutorial workflow and supplies checked snippets to the
served documentation. Its calculation and rendering steps therefore remain visible in one ordered
script body.

The complete Sugarcane and Tobacco applications have a different purpose. Their scientific
workflows are already established, but their repeated Matplotlib construction and PDF-writing code
made the top-level analytical sequence difficult to scan. Tobacco was especially affected because
its two deterministic response-page loops occupied much of the numbered script.

The package plotting layer and hidden report helpers were removed deliberately. Improving local
program structure must not recreate either abstraction.

## Decision

Examples 06 and 07 use a `main()` function for scientific computation and orchestration. `main()`
continues to own:

- direct `X.csv` and `Y.csv` reading;
- physical predictor-coordinate and response-page construction;
- component-path evaluation and model selection;
- fixed-model fitting and cross-validated prediction;
- immutable inspection-result calculation;
- explicit calls that choose which final reports are written.

Private functions defined in the same numbered example own Matplotlib figure construction, layout,
file writing, and figure closing. Tobacco's two `PdfPages` loops remain in example 07, but they may
reside in private same-file report-writing functions. Completed public result objects, coordinates,
labels, selection metadata, pagination, and output paths are passed explicitly into those functions.

Pulp remains sequential because it is a checked tutorial source. Rendering functions must not move
into `pipls`, `examples/_support`, or a shared Sugarcane/Tobacco reporting framework. They must not
read datasets, fit estimators, run cross-validation, select models, or calculate inspection results.

Structural tests protect this computation/rendering boundary without pinning exact private function
names, signatures, source order, or Matplotlib call counts.

This decision supersedes Decision 0069 only where it required Tobacco pagination loops and PDF
operations to remain top-level inline. It clarifies Decision 0083: chart construction may be grouped
in private functions when those functions remain directly inspectable in the numbered caller script.
Decisions 0067, 0079--0083, 0118, and 0127 otherwise remain in force.

## Consequences

- Sugarcane and Tobacco expose compact, readable scientific sequences in `main()`.
- Rendering remains caller-owned and locally inspectable.
- Tobacco preserves full-SVD selection, one-standard-error selection, decreasing wavenumbers,
  source-order pagination, raw observation diagnostics, and the existing five-file PDF inventory.
- Pulp remains the sequential tutorial reference.
- No package API, numerical calculation, output filename, page grouping, or visual content changes.

## Subsequent refinement

Decision 0141 adds `predictor_rank_profile.pdf` to both spectral workflows. It supersedes only
the five-file artifact inventory stated above; the same-file computation/rendering boundary
remains unchanged.
