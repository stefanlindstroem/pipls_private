# Model inspection and post-analysis contract

## Purpose

This document is the durable maintainer contract for fitted-model interpretation, prediction
diagnostics, plotting, and analysis artifacts. Read it before changing `pipls.inspection`,
`pipls.plotting`, post-analysis example helpers, or the real-data analysis reports.

Decisions 0042 and 0043 establish the original architecture and mathematical plotting contracts.
Decision 0045 corrects the ownership boundary between comparison models, Pi-PLS-specific
factorization inspection, and shared PLS-family analysis. The migration is complete: shared
latent-structure functions use estimator-neutral names and structural fitted-model contracts, and
numbered-example post-analysis applies them only to the selected Pi-PLS model. The Pi-PLS $P$, $D$,
and $Q$ surface remains explicitly method-specific.

## Analysis stages

Keep these stages separate in code, documentation, filenames, and figure labels.

### Model-selection diagnostics

The component-path workflows answer how cross-validated loss changes with component count and, for
Pi-PLS, the conditional predictor rank. Example 09 owns the canonical Pi-PLS and ordinary PLS path
CSVs and their overlaid comparison PDFs. Examples 10–12 own one Pi-PLS `component_path.csv` and
`component_path.pdf` beside each dataset post-analysis report.
`examples/_support/pls_component_path.py` remains comparison-specific;
`examples/_support/plot_component_path.py` renders both single Pi-PLS paths and explicit
comparisons from canonical CSV files.

A component-path result is not a fitted-model interpretation and is not an unbiased estimate of a
subsequent user choice made after inspecting that same path. Fold SD is descriptive only, and the
numbered examples keep component count as a visible user choice.

### Fixed-model interpretation

A fitted model may be interpreted through its decomposition, scores, loadings, and coefficients.
Unless a separate resampling construction is stated, these quantities describe a model fitted on
all supplied observations. Figure text and documentation must not present them as validation
results.

### Prediction diagnostics

Prediction diagnostics receive `y_true` and `y_pred` explicitly. They must record one precise
prediction kind:

- `fitted values`;
- `fixed-parameter OOF predictions`;
- `selection-conditioned OOF predictions`;
- `external test predictions`.

Use another label only when it is equally explicit and covered by the same validation rules. Do not
use the generic label `cross-validated predictions` when the parameters were selected using the
same observations.

## Package and example ownership

### `pipls.inspection`

The inspection submodule owns reusable numerical computations and immutable result objects. It
uses NumPy and package/scikit-learn fitted results, but not pandas or Matplotlib. Returned arrays
must be defensive copies and read-only where practical.

Implemented responsibilities are:

- canonical Pi-PLS display factors;
- standardized prediction diagnostics;
- shared PLS-family latent-structure extraction;
- shared PLS-family observation diagnostics;
- balanced two-component score-loading biplot coordinates.

The estimator-neutral names and structural fitted-model validation required by Decision 0045 are
implemented. Compatible `PiPLSRegression` and `PLSRegression` models are covered by focused tests.

### `pipls.plotting`

The plotting submodule owns reusable Matplotlib figures for computed analysis results. It imports
Matplotlib inside plotting code so that importing `pipls` and `pipls.inspection` does not require
the optional plotting dependency.

Plotting functions return the `Figure` and named axes. They do not call `show()`, write files,
retain models, or change supplied arrays. The names remain under `pipls.plotting`; they are not
added automatically to `pipls.__all__`.

### `examples/`

`examples/01_minimal_fit_and_plot.py` is the primary onboarding path: literal NumPy matrices, one
fixed fit, one prediction call, and one decomposition plot. It must not depend on the complete
workflow helpers, pandas, cross-validation, or parameter selection.

The complete real-data examples own scientific orchestration:

- direct dataset reading and alignment;
- visible fixed component and predictor-rank choices;
- explicit cross-validation splitters;
- fixed-parameter OOF loops;
- pandas table construction;
- canonical CSV writing;
- CSV rereading and multipage PDF composition;
- dataset-specific physical-axis labels, response subsets, and pagination.

Their support modules live under `examples/_support/` so they are visibly separate from numbered
user entry points. Do not hide real-data reading behind a package loader. Do not place user analysis
helpers under `scripts/`, which remains the repository-maintenance and preparation area.

Numbered examples are pedagogical scripts, not production applications. Each one must demonstrate a
recognizable minimal use case, explicit comparison, or focused benchmark and must be understandable
without knowledge of a paper, manuscript, or project history. State what the data represent and
label printed values or generated artifacts so their meaning is clear. An isolated collection of API
features without a coherent problem belongs in documentation or focused tests, not in a numbered
example.

Keep genuine user choices named, but inline one-use arguments whose function names already explain
them. Trust committed CSV headers and the tracked `examples/results/` directory structure; do not add
repeated dtype, ordering, missing-value, or directory-creation checks to the scripts. Reusable helper
contracts remain tested separately. Brevity removes scaffolding, not explanatory context.

Variable-name acquisition is also example-owned. When a table has meaningful headers, examples
should derive names visibly from `X.columns` and `Y.columns` and pass them to plotting functions.
Users without header-bearing tables may supply names from any explicit metadata source. Package
plotting code must neither read files nor generate scientific-looking fallback names.

## Pi-PLS factorization display

For a fitted standardized map,

\begin{equation}
B_{\mathrm{cs}} = P D Q^\mathsf{T}.
\end{equation}

The primary decomposition report displays:

- predictor rotations $P$;
- dilation values $d_k=D_{kk}$;
- dilation-weighted response rotations $d_kq_{jk}$, stored or plotted as $QD$.

Use the terms **predictor rotation** or **predictor direction** for $P$. Do not call $P$ an X
loading: `PiPLSRegression.x_loadings_` is a separate score-reconstruction quantity.

### Display signs

Signs may be canonicalized only on copies. For component $k$, locate the largest-magnitude entry of
$P_{:k}$. Choose a sign that makes that entry nonnegative, and apply the same sign to $Q_{:k}$.
This must preserve

\begin{equation}
P_{\mathrm{display}}D Q_{\mathrm{display}}^\mathsf{T}=PDQ^\mathsf{T}.
\end{equation}

Ties must be resolved deterministically by the first array position. Zero columns remain unchanged.
Never modify `model.decomposition_`, `x_rotations_`, `y_rotations_`, or another fitted array.

### Predictor rendering

The caller chooses an explicit predictor style:

- `bar` for a small unordered or categorically named predictor set;
- `line` for an ordered physical coordinate.

For categorical bar plots, predictor and response labels are required. Real-data examples pass the
column headers read from `X.csv` and `Y.csv`; plotting code must not replace scientific variable
names with generated labels such as `x1` or `y1`. For line plots, the caller supplies the coordinate
values and axis label. Preserve the supplied order, including a decreasing wavenumber axis. Do not
infer spectra from feature count or names. Do not smooth, interpolate, or normalize plotted vectors
unless a separate documented computation produced that result.

Selected components share one axis per plotted quantity. Categorical values use grouped bars, with
component bars side by side within each predictor or response. Ordered predictor quantities use one
line per selected component on the same physical axis. Do not create one subplot per component when
the components represent directly comparable values on the same variables.

## Prediction diagnostics

Residuals follow

\begin{equation}
e_{ij}=y_{ij}-\hat y_{ij}.
\end{equation}

Standardization for a common display uses the observed responses supplied to the diagnostic:

\begin{equation}
\bar y_j=\frac{1}{n}\sum_{i=1}^n y_{ij},\qquad
s_j=\sqrt{\frac{1}{n-1}\sum_{i=1}^n (y_{ij}-\bar y_j)^2},
\end{equation}

and

\begin{equation}
z_{ij}=\frac{y_{ij}-\bar y_j}{s_j},\qquad
\hat z_{ij}=\frac{\hat y_{ij}-\bar y_j}{s_j},\qquad
e^{(z)}_{ij}=z_{ij}-\hat z_{ij}.
\end{equation}

The scale therefore uses `ddof=1`. The same observed-response center and scale apply to predictions.
Reject nonfinite inputs, shape disagreement, fewer than two observations, and constant response
columns.

The initial figure contract contains:

- standardized observed versus predicted values with an identity line;
- standardized residuals versus standardized predictions with a zero line;
- response-wise standardized RMSE.

A pooled residual histogram and fitted normal density are excluded. They combine responses that may
have different error structures and add a distributional display not required by the diagnostic.

## Shared PLS-family analysis

Scores, reconstruction loadings, regression coefficients, balanced score-loading biplots, and raw
score-distance or X-reconstruction-residual diagnostics are shared PLS-family analyses. They are not
specific to ordinary `PLSRegression`, and their final reusable API names must not contain either
`pls` or `pipls`.

The shared numerical API may operate on a fitted `PLSRegression` or `PiPLSRegression` when the
object exposes compatible public attributes and transformations. Latent-structure extraction uses:

- `x_scores_`;
- `x_loadings_`;
- `y_loadings_`;
- `coef_`.

Observation diagnostics additionally require callable `transform()` and `inverse_transform()`
methods. The implementation validates this structural contract rather than using a concrete
`isinstance(..., PLSRegression)` restriction. Coefficients retain the public orientation
`(n_targets, n_features)`.

Example 09 uses ordinary PLS only for the comparative component-path CV-MSE curves. Examples
10–12 evaluate Pi-PLS paths only. After a Pi-PLS configuration is selected, one fitted
`PiPLSRegression` supplies the shared
scores, loadings, coefficients, biplot coordinates, observation diagnostics, and OOF prediction
diagnostics. No second ordinary PLS model is fitted for post-analysis. Tests may and should apply
the shared API to both estimator classes.

Component and response subsets are explicit function arguments. Selected X or Y loading components
share one axis, using grouped bars for named categorical variables or overlaid lines for a physical
predictor axis. Selected coefficient responses likewise share one axis. Do not choose responses by
hidden heuristics. Do not draw thousands of biplot arrows for spectral data.

For a supplied observation with X score $t_i$, let $\bar t_{\mathrm{train}}$ and
$S_{T,\mathrm{train}}$ be the center and sample covariance of the fitted training scores. The raw
score distance is

\begin{equation}
h_i=(t_i-\bar t_{\mathrm{train}})^\mathsf{T}S_{T,\mathrm{train}}^{+}(t_i-\bar t_{\mathrm{train}}).
\end{equation}

Here $+$ denotes the Moore--Penrose inverse. The X reconstruction residual is

\begin{equation}
q_i=\lVert x_i-\hat x_i\rVert_2^2.
\end{equation}

The reconstruction $\hat x_i$ is obtained through the fitted model's public
transform/inverse-transform round trip. These are descriptive raw quantities. Do not add
theoretical probability limits, automatic outlier labels, or contribution plots without a separate
decision.

For selected $t_k$ and $p_k$, the biplot uses
$a_k=\sqrt{\lVert p_k\rVert_2/\lVert t_k\rVert_2}$,
$\tilde t_k=a_kt_k$, and $\tilde p_k=p_k/a_k$. Tests must preserve
$\tilde T\tilde P^\mathsf{T}=T_{\mathcal K}P_{\mathcal K}^\mathsf{T}$ and equal component-wise
score/loading norms. The biplot is enabled only for Pulp.

VIP, automatic variable selection, confidence ellipses, uncertainty intervals, permutation tests,
contribution plots, and theoretical outlier thresholds require separate decisions.

## Artifact contract

For real-data post-analysis, CSV is canonical and PDF is a derived view. Physical predictor
coordinates are caller-owned metadata read during example input handling and passed explicitly to
the report renderer; they are not inferred by `pipls.plotting`. The example writes numeric
tables, reads them back, and constructs the report from those reread tables.

Prediction tables use long form and retain at least:

```text
sample
fold
response
observed
predicted
residual
observed_standardized
predicted_standardized
residual_standardized
prediction_kind
```

Pi-PLS response-direction tables retain at least:

```text
response
component
q
dilation
weighted_q
```

Predictor-direction, score, loading, coefficient, and observation-diagnostic tables use explicit
sample, feature, response, and component identifiers as applicable. Do not serialize estimators as
part of the result contract.

The seven common tables are required for every complete post-analysis workflow:

```text
pipls_predictor_directions.csv
pipls_response_directions.csv
predictions.csv
x_scores.csv
x_loadings.csv
y_loadings.csv
coefficients.csv
```

A workflow may add `observation_diagnostics.csv` with columns `sample`, `score_distance`, and
`x_reconstruction_residual`. Tobacco uses this optional eighth table. Response pagination must
partition the source response names exactly once and preserve their source order.

A figure page must identify the dataset, model, selected components or responses, and prediction
kind where predictions are shown. The example-level report composer controls page order and
pagination; a package plotting function renders one explicit selection at a time.

The Pulp workflow uses the same five non-shuffled folds as its component-path comparison. Under
Decision 0045 it clones only the already fixed Pi-PLS estimator in each fold. Because the Pi-PLS
component count and predictor rank were chosen after inspecting paths computed from the same 46
observations, the resulting OOF predictions are selection-conditioned rather than independent
validation. The standardized values stored in
`predictions.csv` use the full supplied observed-response means and sample standard deviations
for display; they do not reproduce the fold-local scaling used by the component-path loss.

## Testing boundary

Pure inspection tests should verify equations, shapes, finite-value validation, defensive copying,
read-only results, sign preservation, and no estimator mutation.

Plot tests should use a headless Matplotlib backend and verify returned figures, named axes, line and
bar modes, label validation, and successful PDF rendering. Do not freeze pixel values, exact artist
counts unrelated to the contract, or Matplotlib implementation details.

Example-helper tests should use small synthetic tables. They may freeze canonical CSV column names,
prediction provenance, sample order, and PDF generation from reread tables. They must not execute
the complete Pulp, Sugarcane, or Tobacco analyses in `make check`; those remain under
`make examples`.

## Implementation order

The accepted order after Decision 0042 is:

1. pure Pi-PLS display-factor and prediction-diagnostic computations — **complete**;
2. Pi-PLS decomposition and prediction plotting — **complete**;
3. estimator-neutral shared PLS-family analysis — **implemented**;
4. estimator-neutral shared inspection and plotting API — **complete**;
5. Pi-PLS-only Pulp, Sugarcane, and Tobacco post-analysis migration — **complete**;
6. stale-name, artifact, documentation, and boundary-test cleanup — **complete**;
7. return to product documentation and release hardening — **next**.
