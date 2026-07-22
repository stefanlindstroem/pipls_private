# Model inspection and post-analysis contract

## Purpose

This document is the durable maintainer contract for fitted-model interpretation, prediction
diagnostics, plotting, and analysis artifacts. Read it before changing `pipls.inspection`,
`pipls.plotting`, or the real-data analysis workflows.

Decisions 0042 and 0043 establish the original architecture and mathematical plotting contracts.
Decision 0045 corrects the ownership boundary between comparison models, Pi-PLS-specific
factorization inspection, and shared PLS-family analysis. Decision 0058 establishes one chart per
public plotting function and caller-owned figure composition. Decisions 0067–0070 make all four
real-data workflows direct in-memory analyses. The shared latent-structure functions use
estimator-neutral names and structural fitted-model contracts, and numbered-example post-analysis
applies them only to the selected Pi-PLS model. The Pi-PLS $P$, $D$, and $Q$ surface remains
explicitly method-specific.

## Analysis stages

Keep these stages separate in code, documentation, filenames, and figure labels.

### Model-selection diagnostics

The component-path workflows answer how cross-validated loss changes with component count and, for
Pi-PLS, the conditional predictor rank. Example 09 owns the overlaid Pi-PLS and ordinary-PLS
comparison PDFs and plots both immutable paths directly in memory. Pulp, Sugarcane, and Tobacco plot
`component_path_` directly in the numbered examples and write final component-path PDFs without CSV
conversion. Pulp also plots the evaluated predictor-rank profile at its chosen component count.
`examples/_support/pls_component_path.py` remains only for the nontrivial ordinary-PLS fold-local
path calculation; there is no comparison plotting helper.

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

A public single-chart plotting function draws exactly one scientific chart on exactly one
Matplotlib `Axes`. It accepts `ax=None` and returns `(figure, axis)`. With no axis it creates one
figure containing one axis. With a supplied axis it draws without clearing the axis, changing the
figure layout, or creating another figure. `figsize` applies only to standalone creation and is
rejected together with `ax`.

The plotting function owns the chart geometry and may provide concise semantic axis labels and an
axis title. The caller may replace or remove them through the returned axis. Multi-series artists
carry labels, but the plotting function does not create a legend; callers and examples own legend
creation, placement, and styling. Callers also own subplot grids, mosaics, figure-level titles,
layout adjustment, file writing, display, and closing.

Decision 0059 replaces the Pi-PLS decomposition composite with separate one-axis plots for $P$,
$D$, $Q$, and $QD$. Decision 0060 likewise splits prediction diagnostics into observed-versus-
predicted, residual-versus-predicted, and standardized-RMSE charts. Decision 0061 completes the
boundary: the example layer creates every report figure and axis and passes `ax` explicitly to each
package plotter. All public plotters follow the single-axis contract. They do not create legends or
panels, call `show()`, write files, retain models, or change supplied arrays. The names remain under
`pipls.plotting`; they are not added automatically to `pipls.__all__`.

### `examples/`

`examples/01_minimal_fit_and_plot.py` is the primary onboarding path: literal NumPy matrices, one
fixed fit, one prediction call, and one caller-composed panel of the four Pi-PLS factor plots. It
must not depend on the complete workflow helpers, pandas, cross-validation, or parameter selection.

The complete real-data examples own scientific orchestration:

- direct dataset reading and alignment;
- visible fixed component and predictor-rank choices;
- explicit cross-validation splitters;
- fixed-parameter OOF prediction;
- immutable inspection-result construction;
- explicit figure, axis, legend, title, saving, and closing operations;
- dataset-specific physical-axis labels, response subsets, and pagination.

Pulp, Sugarcane, and Tobacco keep those stages directly in their numbered scripts and use pandas
only for committed input files. Do not hide real-data reading behind a package loader.
Do not place user analysis helpers under `scripts/`, which remains the repository-maintenance and
preparation area.

The served Pulp tutorial is the primary pedagogical analysis. It extracts executable snippets from
that workflow and presents each deterministic generated figure separately. Plot-specific sections
link to the general inspection and plotting references rather than duplicating API contracts.
Prediction figures must retain the selection-conditioned OOF provenance.

Documentation ownership is fixed by Decision 0065. The tutorial owns the linear Pulp narrative and
all generated Pulp figures. `model_inspection.md` owns general figure-by-figure interpretation with
stable anchors. `docs/api/plotting.md` owns signatures and operational contracts only. Example pages
may describe script purpose and artifacts but must not reproduce the tutorial analysis or embed its
generated figures.

Pulp is the canonical tutorial analysis. `examples/10_pulp_real_data.py` owns its direct pandas
loading, `PiPLSPathCV(refit=False)` evaluation, visible three-component choice, conditional
predictor-rank profile, fixed `PiPLSRegression` fit, five-fold scikit-learn OOF predictions,
inspection computations, and final PDF composition. The selected component row is retrieved before
the selection figures, while the fixed estimator is fitted only after those figures have been
constructed. It intentionally adds no external scaler because `PiPLSRegression` learns predictor
and response standardization inside each fit.

The tutorial extracts checked snippets directly from example 10. `tools/render_pulp_tutorial.py`
repeats the small in-memory numerical sequence rather than importing or executing the
artifact-writing example. It owns tutorial-specific figure dimensions, titles, legends, selected
display components and responses, SVG writing, closing, and the generated manifest. It must call the
public one-axis plotters rather than reproduce plotting logic. Generated tutorial files remain
derived documentation assets and are not committed.

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

Numbered complete-analysis examples keep immutable numerical results in memory and generate only
final PDF figures. Physical predictor coordinates remain caller-owned and are never inferred by
`pipls.plotting`.

Pulp writes six one-page PDFs because it includes a conditional predictor-rank profile. Sugarcane
writes five one-page PDFs. Tobacco writes five PDFs; its prediction-diagnostic and coefficient files
each contain three deterministic source-order response pages. The first two pages contain five
responses and the last page contains the remaining three.

A figure page must identify the dataset, model, selected components or responses, and prediction kind
where predictions are shown. The numbered example controls pagination, panel geometry, legends,
figure-level titles, PDF writing, and closing; a package plotting function renders one explicit
selection on one supplied axis at a time. Tobacco retains a $2\times2$ factor figure, one
$1\times3$ prediction figure per response page, one $2\times2$ latent/observation figure, and one
full-width coefficient figure per response page.

Pulp, Sugarcane, and Tobacco use five non-shuffled folds through scikit-learn
`cross_val_predict()`. Because component count and predictor rank are chosen after inspecting paths
computed from the same observations, the resulting OOF predictions are selection-conditioned rather
than independent validation. The standardized display values in `PredictionDiagnostics` use the
full supplied observed-response means and sample standard deviations; they do not reproduce the
fold-local scaling used by the component-path loss.

## Testing boundary

Pure inspection tests should verify equations, shapes, finite-value validation, defensive copying,
read-only results, sign preservation, and no estimator mutation.

Plot tests should use a headless Matplotlib backend and verify returned figures and axes, line and
bar modes, label validation, and successful PDF rendering. Structural tests should verify that
public plotters accept `ax`, use only the shared one-axis resolver for standalone creation, and do
not create legends, panels, files, displays, or closing operations. Do not freeze pixel values,
exact artist counts unrelated to the contract, or Matplotlib implementation details.

Structural Pulp, Sugarcane, and Tobacco tests should verify direct `component_path_` access,
scikit-learn OOF prediction, in-memory inspection, explicit `ax=` composition, and final PDF
filenames without executing the artifact-writing scripts. Tobacco tests additionally protect full
predictor SVD, decreasing-wavenumber rendering, source-order response pagination, raw observation
diagnostics, and the two caller-owned multipage PDF loops. A focused Pulp numerical test protects
the selected pair, rank-profile boundary interpretation, OOF dimensions, and inspection alignment.
Complete Pulp, Sugarcane, and Tobacco runs remain under `make examples`.

## Implementation order

The accepted order after Decision 0042 is:

1. pure Pi-PLS display-factor and prediction-diagnostic computations — **complete**;
2. Pi-PLS decomposition and prediction plotting — **complete**;
3. estimator-neutral shared PLS-family analysis — **implemented**;
4. estimator-neutral shared inspection and plotting API — **complete**;
5. Pi-PLS-only Pulp, Sugarcane, and Tobacco post-analysis migration — **complete**;
6. stale-name, artifact, documentation, and boundary-test cleanup — **complete**;
7. immutable component-path result simplification — **complete**;
8. direct Sugarcane in-memory workflow — **complete**;
9. direct Pulp and tutorial workflow — **complete**;
10. direct Tobacco workflow and table-helper removal — **complete**.
11. direct Pi-PLS/ordinary-PLS comparison — **complete**;
12. final result-surface, documentation, and structural-policy cleanup — **complete**.
