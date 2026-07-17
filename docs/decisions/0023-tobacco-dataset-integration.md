# Decision 0023: tobacco dataset integration

## Status

Accepted during Phase E3.

## Context

The public tobacco collection provides 347 raw FT-NIR spectra, sample metadata, and quantitative
reference measurements for 13 chemical components. It is a useful high-dimensional multi-output
regression case, but the repository integration must preserve the transparent programming-user
contract and must not expose private archive paths or preparation-only tooling.

The spectra and chemistry are supplied in separate public workbooks and share a unique sample
identifier. The spectral workbook also contains cultivation-year, geographical-origin, and
spectrum-index metadata that are not predictors for the regression example.

## Decision

- Integrate Mendeley Data DOI `10.17632/9z7dgdtggk.1` as the fourth repository real dataset under
  CC BY 4.0.
- Match the two public tables one-to-one by their shared sample ID and order the result by that ID.
- Retain all 347 matched samples, all 1,557 raw spectral columns, and all 13 chemical responses.
- Exclude the sample ID, cultivation year, geographical origin, and duplicate spectrum-index
  columns from the model matrices.
- Apply no imputation, smoothing, derivatives, scatter correction, centering, scaling, or other
  spectral preprocessing.
- Publish only `X.csv`, `Y.csv`, public metadata, attribution, and a direct-reading example; do not
  add a package loader or preparation-only script.
- Treat the initial Phase E3 integration suite as complete after Linnerud, pulp, sugarcane, and
  tobacco; proceed next to benchmark-fixture design rather than adding datasets indefinitely.

## Consequences

- The repository gains a 347-by-1,557 spectral predictor matrix with 13 responses and public,
  redistributable provenance.
- Users see the ordinary `read X`, `read Y`, and fit workflow without hidden data utilities.
- The committed matrices represent raw spectra; any future preprocessing must be explicit and
  fold-safe in the analysis code.
- Phase E4 may now define comparative benchmark manifests across four structurally distinct real
  datasets without claiming manuscript reproduction.
