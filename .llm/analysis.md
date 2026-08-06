# Model inspection and post-analysis contract

## Purpose

This file defines how fitted Pi-PLS results are interpreted after model selection. Analysis is
separate from search, full-data fitting, and prediction provenance. Numerical computations belong
in `pipls.inspection`; figure composition and saving belong to callers.

## Analysis stages

Use the stages in this order.

### 1. Selection diagnostics

Inspect cross-validated evidence before fitting the final model:

- `search.component_path_` for the component path;
- `search.predictor_rank_profile(n_components)` for the conditional rank profile;
- `search.select(...)` when a fitting-free selected row is needed.

Decision 0148 makes this route explicitly hierarchical. The predictor-rank profile distinguishes
the exact configured-score reference from the smaller tolerance-qualified retained rank. The
component path then contains only those conditionally retained rows, and component-count rules act
on that path. Candidate evidence in `cv_results_` remains broader than named-rule eligibility.

CV-MSE plots show arithmetic mean plus or minus population SD across materialized validation
splits. SD is descriptive variability, not a confidence interval and not a selection threshold.
Tolerance thresholds come from `PiPLSSelection.cv_mse_threshold`.

### 2. Fixed-model interpretation

Fit the exact selected row with `search.refit(..., selection=selection)` or fit
`PiPLSRegression` directly. Compact workflows may instead resolve a rule or component count inside
`refit()`. Then inspect:

- Pi-PLS-specific predictor directions, dilation, response directions, and weighted response
  directions;
- shared PLS-family scores, loadings, coefficients, balanced biplot coordinates, and observation
  diagnostics.

Interpretation is conditional on the fitted model and training data. It is not validation.

### 3. Prediction diagnostics

Predictions are supplied explicitly to `prediction_diagnostics()`. The caller states whether they
are fitted, OOF, or external-test predictions through `PredictionKind`. Diagnostics must never infer
provenance from array shape or estimator state.

OOF predictions come from `search.oof_report(..., selection=...)` and remain selection-conditioned.
When the same fitted search informed the selection, those predictions support review of that
selection but do not independently qualify or validate it. External-test predictions require an
independently held-out matrix supplied by the user.

## Package ownership

`pipls.inspection` owns pure numerical transformations and immutable records. It must not:

- import Matplotlib or `adjustText`;
- create or save figures;
- choose components, responses, labels, colors, page layouts, or output paths;
- read datasets or infer sample identity;
- perform model selection or full-data fitting.

All returned arrays are defensive, read-only, finite, directly validated, and pickle-safe.
Unrepresentable derived quantities raise clear exceptions instead of returning silent infinities or
NaNs.

## Pi-PLS display factors

`pipls_display_factors(decomposition, ...)` accepts `PiPLSDecomposition` and returns
`PiPLSDisplayFactors` containing:

- predictor directions `P`;
- nonnegative dilation vector `d`;
- response directions `Q`;
- derived weighted response directions `Q D`.

Default display signs make the first largest-magnitude predictor-direction entry positive for each
component. A response anchor may instead orient one response entry positive or negative. Exact zero
response anchors fall back to the predictor convention.

Sign changes are display transformations only and must preserve:

```text
P D Q.T
```

The result stores numerical arrays, not labels or sign bookkeeping.

## Shared latent structure

`latent_structure(model)` accepts a compatible fitted PLS-family estimator through public
`x_scores_`, `x_loadings_`, `y_loadings_`, and `coef_` attributes. Both Pi-PLS and scikit-learn
`PLSRegression` satisfy this numerical boundary.

`LatentStructure` contains defensive read-only copies. Coefficients retain the common
`(n_targets, n_features)` orientation. Shared analysis names are estimator-neutral; Pi-PLS-specific
factorization names retain the `pipls` marker.

Numbered real-data examples apply shared fitted-model interpretation only to their selected Pi-PLS
model. Ordinary PLS is fitted in the dedicated path-comparison example, not as a second final model
in examples 05--07.

## Balanced biplot coordinates

`biplot_coordinates(structure, components=(i, j))` accepts exactly two distinct zero-based
components. It balances sample score and predictor loading coordinates by reciprocal positive
scales while preserving the selected reconstruction:

```text
T_selected @ P_selected.T
```

The result contains sample coordinates, predictor coordinates, selected component indices, and the
actual scaling factors. Labels and arrow geometry remain caller-owned.

## Observation diagnostics

`observation_diagnostics(model, X)` uses public transform and inverse-transform behavior plus fitted
training scores and loadings. It returns:

- squared score distance relative to the fitted training-score covariance;
- row-wise squared predictor reconstruction residual.

These are raw diagnostic quantities. The package supplies no theoretical probability limits,
outlier classification, or automatic exclusion.

## Prediction diagnostics

`prediction_diagnostics(observed, predicted, kind=...)` accepts one- or two-dimensional responses
and normalizes them to a two-dimensional numerical result. It derives:

- residuals `observed - predicted`;
- observed-response centers;
- safe observed-response sample scales with `ddof=1`;
- standardized observed values, predictions, and residuals;
- response-wise standardized RMSE.

The result stores only independent observed values, predicted values, and provenance as constructor
inputs; derived arrays are recomputed and validated. Constant response columns and ambiguous
provenance are rejected.

## Rendering boundary

Examples and users render immutable arrays directly with Matplotlib. They own:

- figure and axes construction;
- component/response choice;
- physical coordinate axes such as wavelength or wavenumber;
- labels, annotations, legends, styles, and panel layout;
- optional `adjustText` placement;
- saving and closing artifacts.

The runtime package exposes no `pipls.plotting` module, no `plot_*` helpers, and no artist-bearing
result records. Matplotlib and `adjustText` remain optional example/documentation dependencies.

Annotated biplots call `adjust_text()` only after axis configuration. Tests may protect this order
and direct ownership but must not freeze final adjusted label coordinates.

## Maintained workflow roles

- Example 01: fitted-value prediction diagnostic only; no OOF claim.
- Example 02: selection evidence, selection-driven fixed refit, and external-test prediction.
- Example 03: selection-only leave-one-out report with ordered OOF predictions.
- Example 04: Pi-PLS and ordinary-PLS path comparison only.
- Example 05: repeated-CV Pulp selection, matching OOF report, selection-driven refit, and
  representative interpretation.
- Example 06: Sugarcane selection, OOF report, selection-driven refit, spectral interpretation, and
  rank profile.
- Example 07: Tobacco with separately labeled 10% predictor-rank and component-count tolerance
  decisions, OOF report, selection-driven refit, full-SVD spectral analysis, rank profile, raw
  observation diagnostics, and source-order response pagination.

Pulp, Sugarcane, and Tobacco keep scientific computation in memory and write only final PDF
figures. Sugarcane and Tobacco keep analysis in `main()` and group rendering in private functions in
the same script. Reusable numerical logic belongs in the package; dataset-specific report layout
does not.

## Computational-performance documentation boundary

Decision 0150 owns one central served reference page for training cost. It must distinguish changes
to validation evidence, candidate coverage or model policy, numerical approximation, parallel wall
time, and repeated OOF work. It may describe candidate-fit scaling and current execution boundaries,
but it must not publish machine-independent speed claims or imply that narrower validation or search
policies are statistically equivalent.

The guide uses candidate-fold fits as its first accounting unit. For $N_{\mathrm{pair}}$ evaluated
pairs and $N_{\mathrm{split}}$ materialized splits, candidate evaluation performs
$N_{\mathrm{pair}}N_{\mathrm{split}}$ fold-local fits, while the feasibility pass adds one Pi-PLS
probe per split. Full-data refitting adds one fit and each OOF report adds one selected-pair fit per
split. These counts are not presented as wall-clock formulas.

All advice preserves fold-local learned preprocessing. Shuffled validation and randomized predictor
SVD examples use explicit integer seeds. Fixed and maximum-rank examples omit `search_method`;
`"adaptive"` and `"exhaustive"` are used only when predictor-rank coverage is an actual choice.

## Artifact and testing contract

Generated figures, manifests, and example outputs are not fitted package state and must not be
committed except for deliberate documentation assets. Tests should protect:

- inspection equations, shapes, finiteness, immutability, direct construction, and pickle behavior;
- exact preservation identities for display signs and biplot balancing;
- explicit prediction provenance and residual orientation;
- direct caller-owned rendering and absence of runtime plotting imports;
- deterministic tutorial assets through semantic manifests and parseable SVGs where maintained;
- workflow stage ordering without pinning incidental local variable names or visual coordinates.
