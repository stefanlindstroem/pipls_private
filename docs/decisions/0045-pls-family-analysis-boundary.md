# Decision 0045: separate comparison models from shared PLS-family analysis

## Status

Accepted. Implementation is pending in the corrective series following this decision.

This decision supersedes the ordinary-PLS-specific ownership and example-use portions of Decisions
0042 and 0043. Their separation of selection diagnostics, fitted-model interpretation, prediction
diagnostics, immutable inspection results, optional plotting, explicit labels, physical axes,
canonical CSV artifacts, and mathematically stated biplot scaling remains accepted.

## Context

The real-data examples currently use ordinary scikit-learn `PLSRegression` in two different roles:

1. as the established reference curve in the component-path CV-MSE comparison; and
2. as a second fitted model for OOF predictions, scores, loadings, coefficients, the Pulp biplot,
   and Tobacco observation diagnostics.

The first role is useful and should remain. The second role weakens the intended example narrative.
After model selection, the examples should demonstrate one selected Pi-PLS model and show that the
standard analysis techniques familiar from PLS also apply to that model.

The current package also names shared latent-structure computations and plots after ordinary PLS.
Scores, reconstruction loadings, coefficients, score-loading biplots, and score-distance or
X-reconstruction-residual diagnostics are not unique to `PLSRegression`. Both
`PLSRegression` and `PiPLSRegression` expose the fitted quantities and transformations needed by
these analyses.

Pi-PLS additionally exposes the method-specific factorization

\begin{equation}
B_{\mathrm{cs}}=PDQ^\mathsf{T},
\end{equation}

which requires separate, explicitly Pi-PLS-specific inspection.

## Decision

The repository distinguishes three analysis roles.

### 1. Comparative component-path diagnostics

The Pulp, Sugarcane, and Tobacco component-path figures continue to show Pi-PLS and ordinary PLS
CV-MSE in the same plot. Separate canonical path CSV files remain the source of that figure.

Ordinary `PLSRegression` is permitted in the example-local component-path comparison helper and in
focused benchmarks where it is the declared external comparator. This use answers a comparative
model-selection or validation question; it does not create the model interpreted by the subsequent
post-analysis report.

### 2. Pi-PLS-specific factorization inspection

Inspection of $P$, $D$, $Q$, $QD$, and the identity $PDQ^\mathsf{T}$ is specific to Pi-PLS.
These numerical and plotting names retain an explicit `pipls` marker, including
`pipls_display_factors()` and `plot_pipls_decomposition()`.

The terminology remains predictor rotations or predictor directions for $P$, dilation values for
$D$, and response rotations or dilation-weighted response directions for $Q$ and $QD$. These
quantities must not be presented as ordinary PLS loadings.

### 3. Shared PLS-family analysis

The following analyses are shared PLS-family techniques rather than Pi-PLS-specific techniques:

- X-score plots;
- X-loading plots;
- Y-loading plots;
- regression-coefficient plots;
- balanced score-loading biplots;
- score-distance and X-reconstruction-residual diagnostics;
- observed-versus-predicted and residual diagnostics.

Their reusable numerical and plotting APIs must use estimator-neutral names. They may accept either
a fitted `PLSRegression` or a fitted `PiPLSRegression` when the object exposes the required public
fitted attributes and transformations. The API must validate that structural contract rather than
restricting use through a concrete `isinstance(..., PLSRegression)` check.

The numbered examples apply these shared tools only to the selected `PiPLSRegression` model. After
the comparative component-path figure, they must not fit a second ordinary PLS model for OOF
prediction or interpretation. This demonstrates that standard PLS-family analyses remain available
for Pi-PLS while keeping one coherent fitted-model narrative.

## Naming and artifact consequences

The corrective implementation will:

- retain Pi-PLS-specific names only for $P$, $D$, and $Q$ inspection;
- replace ordinary-PLS-specific shared inspection and plotting names with estimator-neutral names;
- test the shared numerical API with both `PLSRegression` and `PiPLSRegression`;
- remove final ordinary PLS fits and ordinary PLS OOF predictions from numbered examples;
- retain ordinary PLS only in component-path comparison and declared comparator benchmarks;
- rename shared post-analysis CSV files so that they describe their quantities rather than an
  ordinary PLS estimator;
- retain explicit Pi-PLS prefixes for factorization-specific artifact files;
- label report pages with the fitted model identity supplied by the example, even when the plotting
  function itself has a generic name.

No compatibility aliases are required for the mistaken intermediate shared-analysis names because
the package remains unreleased at version `0.0.0`.

## Migration sequence

Implementation proceeds in three further patches:

1. generalize and rename the shared inspection and plotting API, with tests against both estimator
   classes;
2. convert Pulp, Sugarcane, and Tobacco post-analysis to one selected Pi-PLS model while preserving
   the two-model CV-MSE comparison;
3. remove stale ordinary-PLS post-analysis terminology, filenames, tests, and documentation, and
   add static repository-boundary enforcement.

Until those patches are complete, the current runtime names and example behavior are transitional
and must not be treated as the final analysis contract.

## Consequences

- Users retain the familiar Pi-PLS-versus-PLS CV-MSE comparison.
- Every post-selection report interprets and diagnoses one selected Pi-PLS model.
- Pi-PLS-specific and shared PLS-family concepts are distinguished by both ownership and naming.
- Shared tools remain useful to ordinary PLS users without making ordinary PLS the subject of the
  numbered Pi-PLS examples.
- The estimator, numerical core, path search, CV-MSE definitions, and existing factorization
  identities are unchanged by this decision.
