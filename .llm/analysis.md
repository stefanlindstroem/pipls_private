# Model inspection and post-analysis contract

## Purpose

This document is the durable maintainer contract for fitted-model interpretation, prediction
diagnostics, rendering, and analysis artifacts. Read it before changing `pipls.inspection` or the
real-data analysis workflows.

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
Pi-PLS, the conditional predictor rank. Example 04 owns the overlaid Pi-PLS and ordinary-PLS
comparison PDFs and plots both immutable paths directly in memory. Pulp, Sugarcane, and Tobacco plot
`component_path_` directly in the numbered examples and write final component-path PDFs without CSV
conversion. Pulp also plots an immutable predictor-rank profile derived on demand at its chosen
component count.
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

### Rendering boundary

The package owns no plotting submodule and exposes no public `plot_*` convenience functions.
Immutable numerical inspection objects are the durable
interface, and maintained examples render their arrays with ordinary Matplotlib. The example layer
owns chart type, component and response selection, categorical positions, physical coordinates,
labels, reference lines, legends, panel geometry, file writing, and closing.

`biplot_coordinates()` remains package-owned because balancing score and loading coordinates is a
numerical operation. Annotated biplots use optional `adjustText` only after final axis configuration.
The runtime package imports neither Matplotlib nor `adjustText`. Maintained examples and
renderers must not hide chart construction in support helpers.

Maintained rendered figures use `$\Pi$`-PLS when the method name appears. Pi-PLS factor labels use
upper-case `$P$` and `$Q$` for matrix elements and lower-case `$d$` for diagonal elements of `$D$`.
Tiled factor, latent-structure, and prediction-diagnostic figures omit subplot titles; their axis
labels and legends identify the plotted quantities. Prediction-diagnostic figure-level titles stay
on one line. Component-path and predictor-rank-profile figures start at zero and use an upper limit
of at least one.

### `examples/`

`examples/01_pulp_quick_start.py` is the primary onboarding path: package-owned Pulp data,
one chained default path search and one-standard-error refit, one fitted-value prediction call,
standardized response diagnostics, and one caller-composed observed-versus-fitted plot. It must not
depend on complete-workflow helpers, pandas, an explicit splitter, or OOF reporting, and its prose
must identify the predictions as full-data fitted values rather than predictive validation.

The complete real-data examples own scientific orchestration:

- visible dataset acquisition and alignment;
- visible fixed component and predictor-rank choices;
- explicit cross-validation splitters;
- fixed-parameter OOF prediction;
- immutable inspection-result construction;
- explicit figure, axis, legend, title, saving, and closing operations;
- dataset-specific physical-axis labels, response subsets, and pagination.

Pulp, Sugarcane, and Tobacco keep those stages directly in their numbered scripts and obtain their
installed matrices and labels through `load_pulp()`, `load_sugarcane()`, and `load_tobacco()`. The
spectral scripts derive physical coordinates and response names from `PiPLSDataset` while keeping
scientific computation and orchestration in `main()` and rendering in private same-file functions.
Do not introduce a generic real-data loader.
Do not place user analysis helpers under `scripts/`, which remains the repository-maintenance and
preparation area.

Decision 0139 establishes a three-stage served tutorial target. The first stage is a short rendered
Pulp quick start sourced from the renamed `examples/01_pulp_quick_start.py`; it owns the automatic
one-standard-error fit, full-data fitted-value diagnostics, one standardized observed-versus-fitted
axis, and the explicit boundary that calibration fit is not validation. The second stage is the
synthetic tutorial, which retains the fitted search object and owns component-path inspection,
conditional-rank inspection, manual component selection, fixed fitting, and external-test
prediction. The complete Pulp tutorial is the third stage and owns the real-data analysis,
including the direct conditional predictor-rank plot, selection-conditioned OOF diagnostics, and
deterministic inspection figures. It shows
one setup block before use, selects the first three response columns only for pointwise-plot
legibility, and sources each displayed interpretation figure from a matching standalone renderer
block. Clear section headings distinguish estimator-neutral PLS-family latent-structure and
prediction plots from Pi-PLS-specific factorization plots. The tutorial displays both predictor
directions $P$ and weighted response directions $QD$; the complete example retains the separate
$D$ and $Q$ plots. The tutorial omits
raw regression-coefficient visualization because heterogeneous original units make that single
figure unsuitable for the walkthrough; the numbered-example figure remains. Plot-specific
sections link to the general inspection reference rather than duplicating numerical contracts. Prediction figures must retain the selection-conditioned OOF provenance.

Documentation ownership is fixed by Decisions 0074--0078. The API overview owns the public
result-object map, and the task-oriented troubleshooting page owns common public-API recovery
paths without becoming another tutorial. The quick-start tutorial owns the minimum automatic
fitted-model workflow and its calibration-fit boundary. The synthetic tutorial owns
inspect-decide-refit mechanics on controlled data. The Pulp tutorial owns the real-data selection
qualification, selection-conditioned OOF boundary, immutable inspection-result handoff, and a
representative set of generated interpretation figures. The complete plot catalogue belongs to
`model_inspection.md`, while common variations belong to the generated API and advanced guides. The
home page owns a restrained application-oriented motivation, the minimal fixed-fit entry, and
audience routes. The root README owns package orientation, motivation without comparative
performance claims, installation, two compact workflows, and tutorial links; contributor commands
and repository maintenance belong only in `CONTRIBUTING.md`. Tutorial openings present purpose,
coverage, and workflow before source or renderer provenance, which belongs in terminal reproduction
sections. Generated fixed-regression and path
pages own exact
estimator, preprocessing, fitted-state, and result contracts. `path_analysis.md` owns advanced search, splitter, OOF, and validation behavior in one place.
`model_inspection.md` owns the bridge from immutable numerical fields to caller-owned rendering
with stable interpretation anchors and without repeating elementary Matplotlib recipes. Example pages may describe script purpose and artifacts
but must not reproduce the tutorial analysis or embed its generated figures.

Decision 0140 has implemented search-owned selected-row lookup:
`search.select(rule=... or n_components=...)` shares the same private resolver as `refit()`, while
`component_path_` remains aligned numerical evidence.
`search.select(...)` remains the fitting-free selection-only operation; model-producing consumers
obtain the fitted row from `model.selection_`.

Decision 0141 makes every complete real-data workflow inspect the conditional predictor-rank
profile at its selected component count. Sugarcane and Tobacco call
`search.predictor_rank_profile(selection.n_components)`; for Tobacco, `selection` is the exact row
retained by the one-standard-error refit.

Decision 0143 accepts a seven-patch workflow normalization. In the final model-producing examples,
modeling ends when `search.refit(...)` returns the full-data model. Analysis then begins in this
default order:

```python
selection = model.selection_
path = search.component_path_
rank_profile = search.predictor_rank_profile(selection.n_components)
report = search.oof_report(X, Y, selection=selection)
```

`oof_report()` is optional numerical analysis, not a modeling statement. Fitted-model inspection and
all rendering follow these retained search and OOF results. Workflows with an external test set may
omit the OOF report. Validation-only and path-comparison-only examples retain their
specialized roles and need not construct unused final models. All seven patches are complete.
Selection results carry rule and 1-SE reference evidence, refitted models retain `selection_`, and
`oof_report(selection=...)` owns OOF analysis. All
model-producing workflows implement this ordering. The leave-one-out workflow remains model-free,
resolves one `best_score` selection, and passes it to `oof_report()`. The former report API has been
removed without aliases.

Pulp is the canonical manual-selection tutorial analysis. `examples/05_pulp_real_data.py` owns its
public `load_pulp()` acquisition, visible three-component choice, path-evaluating search, full-data
refit, retained `model.selection_`, conditional predictor-rank profile, selection-driven five-fold
OOF report, inspection computations, and final PDF composition. Search and refitting complete
modeling before any selection evidence or diagnostics are retrieved, and every figure is rendered
only after the numerical analysis is complete. It intentionally adds no external scaler because
`PiPLSRegression` learns predictor and response standardization inside each fit.

The tutorial extracts checked snippets directly from example 05. `tools/render_pulp_tutorial.py`
repeats the small in-memory numerical sequence rather than importing or executing the
artifact-writing example. It owns tutorial-specific figure dimensions, titles, legends, selected display components and responses, direct Matplotlib
construction, SVG writing, closing, and the generated manifest. Generated tutorial files remain
derived documentation assets and are not committed.

Decision 0139 Patches 1 and 2 implement the renamed quick-start example, rendered first tutorial,
and three-page served navigation. Its final landing-page and reference reframing is paused while
Decision 0140 introduces search-owned selected-row lookup. The resumed onboarding patch must use
`search.select(...)` rather than teaching path-level scalar selection.

Numbered examples are pedagogical scripts, not production applications. Each one must demonstrate a
recognizable minimal use case or explicit comparison and must be understandable
without knowledge of a paper, manuscript, or project history. State what the data represent and
label printed values or generated artifacts so their meaning is clear. An isolated collection of API
features without a coherent problem belongs in documentation or focused tests, not in a numbered
example.

Keep genuine user choices named, but inline one-use arguments whose function names already explain
them. Trust committed CSV headers and the tracked `examples/results/` directory structure; Git and
source distributions contain only `.gitkeep` placeholders there, while generated PDFs remain local
and excluded from snapshots. Do not add repeated dtype, ordering, missing-value, or
directory-creation checks to the scripts. Reusable helper contracts remain tested separately.
Brevity removes scaffolding, not explanatory context.

Variable-name acquisition is also example-owned. When a table has meaningful headers, examples
should derive names visibly from `X.columns` and `Y.columns` and use them in direct rendering. Users
without header-bearing tables may supply names from any explicit metadata source. The package must
neither read files nor generate scientific-looking fallback names.

## Pi-PLS factorization display

For a fitted standardized map,

\begin{equation}
\mathbf{B}_{\mathrm{cs}} = \mathbf{P}\mathbf{D}\mathbf{Q}^{\mathsf T}.
\end{equation}

The primary decomposition report displays:

- predictor directions $\mathbf{P}$;
- dilation values $d_k=D_{kk}$;
- weighted response directions $d_kQ_{jk}$, exposed or plotted as $\mathbf{Q}\mathbf{D}$.

Use **predictor direction** and **response direction** as the canonical mathematical terms. The
public decomposition fields use `predictor_directions` and `response_directions`; the estimator's
standard PLS-style `x_rotations_` and `y_rotations_` names remain. Do not call $\mathbf{P}$ an X
loading or $\mathbf{Q}$ a Y loading: `PiPLSRegression.x_loadings_` and `y_loadings_` are separate
score-reconstruction quantities. The response-by-mode display $\mathbf{Q}\mathbf{D}$ is the
transpose of the
manuscript's mode-by-response form $\mathbf{D}\mathbf{Q}^{\mathsf T}$.

### Display signs

Signs may be canonicalized only on copies. For component $k$, locate the largest-magnitude entry of
$P_{:k}$. Choose a sign that makes that entry nonnegative, and apply the same sign to $Q_{:k}$.
This must preserve

\begin{equation}
\mathbf{P}_{\mathrm{display}}\mathbf{D}
\mathbf{Q}_{\mathrm{display}}^{\mathsf T}
=
\mathbf{P}\mathbf{D}\mathbf{Q}^{\mathsf T}.
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
columns. Calculate centers, scales, residual standardization, and response-wise RMSE with
range-safe scaled operations. If a required diagnostic cannot be represented as finite float64,
raise a clear `ValueError` rather than returning a nonfinite record.

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

Example 04 uses ordinary PLS only for the comparative component-path CV-MSE curves. Examples
05–07 evaluate Pi-PLS paths only. After a Pi-PLS configuration is selected, one fitted
`PiPLSRegression` supplies the shared
scores, loadings, coefficients, biplot coordinates, observation diagnostics, and OOF prediction
diagnostics. No second ordinary PLS model is fitted for post-analysis. Tests may and should apply
the shared API to both estimator classes.

Component and response subsets are explicit function arguments. Selected X or Y loading components
share one axis, using grouped bars for named categorical variables or overlaid lines for a physical
predictor axis. Selected coefficient responses likewise share one axis. Do not choose responses by
hidden heuristics. Do not draw thousands of biplot arrows for spectral data.

For a supplied observation with X score $t_i$, let $\bar t_{\mathrm{train}}$ and
$\mathbf{S}_{\mathrm{T},\mathrm{train}}$ be the center and sample covariance of the fitted
training scores. The raw score distance is

\begin{equation}
h_i
=
(t_i-\bar t_{\mathrm{train}})^{\mathsf T}
\mathbf{S}_{\mathrm{T},\mathrm{train}}^{+}
(t_i-\bar t_{\mathrm{train}}).
\end{equation}

Here $+$ denotes the Moore--Penrose inverse. The X reconstruction residual is

\begin{equation}
q_i=\lVert x_i-\hat x_i\rVert_2^2.
\end{equation}

The reconstruction $\hat x_i$ is obtained through the fitted model's public
transform/inverse-transform round trip. These are descriptive raw quantities. Do not add
theoretical probability limits, automatic outlier labels, or contribution plots without a separate
decision.

Form the training-score covariance after one common scaling of the centered score matrix, and apply
the same scale to supplied centered scores. The scale cancels in the Moore--Penrose quadratic form
while avoiding overflow in the covariance product. Calculate $q_i$ through a scaled row-wise squared
norm and reject an unrepresentable squared residual.

For selected $t_k$ and $p_k$, the biplot uses
$a_k=\sqrt{\lVert p_k\rVert_2/\lVert t_k\rVert_2}$,
$\tilde t_k=a_kt_k$, and $\tilde p_k=p_k/a_k$. Tests must preserve
$\widetilde{\mathbf{T}}\widetilde{\mathbf{P}}^{\mathsf T}
=\mathbf{T}_{\mathcal K}\mathbf{P}_{\mathcal K}^{\mathsf T}$ and equal component-wise
score/loading norms. Compute norms by max scaling and form $a_k$ as a quotient of square roots
so
finite extreme columns do not overflow before balancing. The biplot is enabled only for Pulp.

VIP, automatic variable selection, confidence ellipses, uncertainty intervals, permutation tests,
contribution plots, and theoretical outlier thresholds require separate decisions.

## Artifact contract

Numbered complete-analysis examples keep immutable numerical results in memory and generate only
final PDF figures. Physical predictor coordinates remain caller-owned and are never inferred by the package.

Pulp and Sugarcane each write six one-page PDFs, including a conditional predictor-rank profile.
Tobacco writes six PDFs; its prediction-diagnostic and coefficient files
each contain three deterministic source-order response pages. The first two pages contain five
responses and the last page contains the remaining three.

A figure page must identify the dataset, model, selected components or responses, and prediction kind
where predictions are shown. The numbered example controls pagination, panel geometry,
legends,
figure-level titles, PDF writing, closing, and each direct Matplotlib selection. These operations may
be grouped in private same-file rendering functions that consume completed numerical results. Tobacco
retains a $2\times2$ factor figure, one $1\times3$ prediction figure per response page,
one
$2\times2$ latent/observation figure, and one full-width coefficient figure per response page.

Pulp, Sugarcane, and Tobacco obtain OOF predictions through explicit
`oof_report(selection=model.selection_)` calls. Every report reuses the exact
`KFold(n_splits=5, shuffle=True, random_state=0)` partition materialized by its path search.
Because component count and predictor rank are chosen after inspecting paths
computed from the same observations, the resulting OOF predictions are selection-conditioned rather
than independent validation. The standardized display values in `PredictionDiagnostics` use the
full supplied observed-response means and sample standard deviations; they do not reproduce the
fold-local scaling used by the component-path loss.

## Testing boundary

Pure inspection tests should verify equations, shapes, direct-construction invariants, finite-value
validation, defensive copying, read-only and pickle reconstruction, sign preservation, no estimator
mutation, representable extreme inputs, and explicit failure for unrepresentable derived values.

Rendering tests should use a headless Matplotlib backend and protect only durable example contracts:
direct access to named immutable arrays, explicit figure and axis construction, physical coordinate
order, expected output files, and successful tutorial SVG generation. Do not freeze pixel values,
automatically adjusted label coordinates, exact artist counts unrelated to the contract, or
Matplotlib implementation details.

Structural Pulp, Sugarcane, and Tobacco tests should verify direct `component_path_` access,
explicit validation reporting, in-memory inspection, direct Matplotlib composition, and final PDF
filenames without executing the artifact-writing scripts. Sugarcane and Tobacco tests should also
verify that analysis remains in `main()` while private same-file rendering functions consume
completed
results. Tobacco tests additionally protect full predictor SVD, decreasing-wavenumber rendering,
source-order response pagination, raw observation diagnostics, and the two multipage PDF loops. A
focused Pulp numerical test protects
the selected pair, rank-profile boundary interpretation, OOF dimensions, and inspection alignment.
Complete Pulp, Sugarcane, and Tobacco runs remain under `make examples`.

## Implementation order

The accepted order after Decision 0042 is:

1. pure Pi-PLS display-factor and prediction-diagnostic computations — **complete**;
2. Pi-PLS decomposition and prediction plotting — **complete**;
3. estimator-neutral shared PLS-family analysis — **implemented**;
4. estimator-neutral shared inspection API and direct-rendering boundary — **complete**;
5. Pi-PLS-only Pulp, Sugarcane, and Tobacco post-analysis migration — **complete**;
6. stale-name, artifact, documentation, and boundary-test cleanup — **complete**;
7. immutable component-path result simplification — **complete**;
8. direct Sugarcane in-memory workflow — **complete**;
9. direct Pulp and tutorial workflow — **complete**;
10. direct Tobacco workflow and table-helper removal — **complete**.
11. direct Pi-PLS/ordinary-PLS comparison — **complete**;
12. final result-surface, documentation, and structural-policy cleanup — **complete**;
13. final data-first rendering policy and structural enforcement — **complete**.


## Public decomposition boundary

`PiPLSDecomposition` is an interpretation result, not a copy of the private construction record. It
exposes the fields `predictor_directions`, `dilation`, and `response_directions`, whose
mathematical values are the predictor directions, mode dilations, and response directions, together
with rank/solver diagnostics and `standardized_regression_map`. The private `PiPLSCoreResult` retains
$\Pi$, $C$, $W$, $P$, $D$,
and $Q$ because numerical invariants and algorithm implementation still require the complete
construction. Downstream studies that need the truncated predictor basis must reconstruct it
without depending on a public intermediate matrix.
## Public fitted-surface cleanup

Decision 0087 distinguishes independent fitted results from exact aliases and execution traces.
`PiPLSRegression` exposes one direction array per side through the decomposition and the standard
PLS-style `x_rotations_` and `y_rotations_` fitted attributes rather than duplicate weight aliases.
Its scorer-specific response scale is private. `PiPLSSearchCV` keeps standard candidate results,
concise immutable path and rank-profile objects, global selection attributes, exhaustive-search qualification, and validation
reporting. A fitted model is returned directly by post-fit `refit()` rather than attached to search
state. Explicit `oof_report(selection=...)` reuses the exact materialized splits and returns OOF
arrays without attaching the report. Validated
input grids, adaptive batches, candidate counters, search-method echoes, and duplicate direct-rank
parameter dictionaries are private implementation details.


## Accepted pre-release public-surface cleanup

Decision 0144 applies one ownership rule to returned numerical records: expose each quantity through
the object that owns it. The seven-patch target removes OOF forwarding properties and impossible
absent-array states, adopts `PiPLSSelection` terminology, removes duplicated fitted-search `best_*`
state and candidate parameter representations, standardizes `PiPLSDataset` on `X` and `Y`, removes
unused shape-only inspection properties, and narrows top-level `pipls` exports to estimators and the
public support warning.

The cleanup retains capabilities with distinct roles. `search.select()` remains the fitting-free
selection operation; all five numerical inspection helpers remain public; balanced biplot scaling
factors remain visible; immutable records retain direct-construction and pickle validation; and
caller-owned Matplotlib rendering remains the plotting boundary. All seven patches are complete:
OOF reports require prediction and count arrays and expose selection metrics only through
`report.selection`; the active API uses `PiPLSSelection` terminology; global-best fitted-search
attributes have been replaced by `search.select(rule="best_score")`; `cv_results_` uses only stable
`n_components` and `predictor_rank` parameter columns; `PiPLSDataset` uses only `X` and `Y`; unused
shape-only inspection properties are removed while component counts remain; and top-level `pipls`
exports only estimators, the public support warning, and version metadata. Result records remain
public from focused modules.

## Accepted final implementation-surface cleanup

Decision 0145 addresses residual implementation and module-boundary duplication after the broader
public-surface cleanup. The four-patch target removes `cv_n_train_min_` and the duplicate
fixed-model
`predictor_rank_`, privatizes `model_selection.py`, removes the unused core-result prediction method
and constant OOF operation-name parameter, declares exact exports for the remaining public estimator
and exception modules, and replaces negative `search.select()` wording with its fitting-free role.

The cleanup retains learned search state, selection provenance, model-selection algorithms, all five
numerical inspection functions, OOF reporting, and caller-owned plotting. All four patches are
complete: the contract is established, redundant fitted attributes are removed, model-selection
algorithms live in `_model_selection.py`, unused private helper layers are gone, remaining public
modules declare exact exports, and tutorial wording states selection ownership positively.

## Accepted CV-MSE tolerance and split-SD transition

Decision 0146 replaces one-standard-error component selection with minimum-CV-MSE selection plus
explicit relative and absolute tolerances. For each component-path row, every materialized
validation split contributes equally to `cv_mse_mean`; `cv_mse_std` is the population SD across the
same split losses and is descriptive only. Maintained figures will show mean CV-MSE ± SD across
validation splits.

A row qualifies only when its mean is below both the relative and absolute thresholds. The first
qualifying component count is selected, retaining the path-owned predictor rank. The default
relative tolerance resolves to `sqrt(float64 epsilon)` and the default absolute tolerance is
positive infinity. Tobacco will demonstrate `relative_tolerance=0.10`; absolute tolerance is
documented but not demonstrated. The complete Pulp workflow and Tutorial 3 will use repeated
five-fold CV with ten repetitions, while quick and other workflows remain lighter.

Current status: **Patch 1 of 7 complete**. This section records the accepted target only; current
source still exposes the one-standard-error and fold-SE surface until the migration patches land.
