# Decision 0042: model inspection and post-analysis architecture

## Status

Accepted. The Pi-PLS and ordinary PLS numerical and plotting foundations are implemented. Pulp,
Sugarcane, and Tobacco are complete real-data post-analysis integrations. The low-dimensional Pulp
biplot and final cross-dataset review are completed by Decision 0043.

## Context

The real-data examples currently answer a model-development question through canonical Pi-PLS and
ordinary PLS component-path CSV files and a CSV-derived comparison PDF. They then fit one fixed
Pi-PLS model chosen through a visible component-count constant. Those artifacts are selection
diagnostics. They do not yet provide a supported analysis of the fitted Pi-PLS factorization,
standard PLS latent structure, predictions, or residuals.

The companion Pi-PLS implementation contains a combined display of the predictor rotations $P$,
the dilation matrix $D$, the response rotations $Q$, standardized observed-versus-predicted values,
and residuals. It also distinguishes line displays for spectra from bar displays for a small set of
scalar variables. These ideas are useful, but the package needs clearer separation between
full-data interpretation and predictive validation, and it must not retain training data merely to
support plotting.

Established PLS analysis also includes scores, loadings, regression coefficients, biplots, and
observation diagnostics. The package should demonstrate a restrained subset of these tools where
they are scientifically readable, without turning plots into automatic variable-selection or
significance procedures.

## Decision

The repository distinguishes three analysis stages:

1. **Model-selection diagnostics.** Component-path CSV files and their comparison PDF describe the
   conditional path over component counts and predictor ranks. The existing
   `examples/pls_component_path.py` and `examples/plot_component_path.py` remain example-local
   helpers for this stage.
2. **Fixed-model interpretation.** Full-data fitted Pi-PLS and ordinary PLS models provide
   decomposition, score, loading, and coefficient quantities. These plots describe the fitted
   models and are not validation results.
3. **Prediction diagnostics.** Observed-versus-predicted and residual displays use predictions
   supplied explicitly by the caller. Every diagnostic records whether the values are fitted,
   fixed-parameter out-of-fold, selection-conditioned out-of-fold, or external-test predictions.

Reusable numerical analysis belongs in a public `pipls.inspection` submodule. It will contain pure
NumPy computations and immutable result objects, with no pandas or Matplotlib dependency. Reusable
plotting belongs in a public `pipls.plotting` submodule. Matplotlib remains optional and is imported
only when that submodule's plotting functions are called. These submodule names are not added to
the top-level `pipls` exports.

Dataset-specific orchestration, explicit data reading, variable-name acquisition, fixed model
choices, out-of-fold prediction loops, table construction, artifact writing, and multipage-report
composition remain under `examples/`. Repository-maintenance and data-preparation code remains
under `scripts/` or the relevant dataset directory; it is not mixed with user analysis.

### Pi-PLS interpretation contract

For the centered and scaled regression map,

\begin{equation}
B_{\mathrm{cs}} = P D Q^\mathsf{T}.
\end{equation}

The primary Pi-PLS decomposition display contains:

- the predictor rotations $P$;
- the dilation values $d_k=D_{kk}$;
- the dilation-weighted response rotations $d_k q_{jk}$, represented by $QD$.

The display calls $P$ predictor rotations or predictor directions, not ordinary PLS loadings.
`PiPLSRegression.x_rotations_` equals $P$, whereas `x_loadings_` is a separately calculated score
reconstruction quantity.

Component signs may be canonicalized on copied arrays for stable display. For each component, the
largest-magnitude entry of the corresponding predictor rotation is made nonnegative, and the same
sign is applied to the response rotation. The transformation must preserve

\begin{equation}
P_{\mathrm{display}}D Q_{\mathrm{display}}^\mathsf{T}=PDQ^\mathsf{T}.
\end{equation}

The fitted estimator and its read-only `decomposition_` arrays must not be mutated.

Predictor rendering is selected explicitly by the example or caller:

- `"bar"` for a small set of scalar predictors;
- `"line"` for an ordered physical axis such as wavelength or wavenumber.

Categorical predictor and response displays require caller-supplied scientific labels. Real-data
examples read `X.csv` and `Y.csv` headers visibly when those headers contain the relevant names; a
programming user may instead supply labels from a schema or any other explicit metadata source.
Selected components share one axis per plotted quantity, with side-by-side bars for categorical
variables and overlaid lines for a physical predictor axis. The plotting API does not read files,
infer whether predictors are spectra, generate substitute variable names, smooth or interpolate
supplied curves, or reorder a supplied physical axis.

### Prediction-diagnostic contract

Prediction diagnostics accept observed and predicted responses explicitly. The estimator does not
retain its training observations for plotting.

Residuals use

\begin{equation}
e_{ij}=y_{ij}-\hat y_{ij}.
\end{equation}

For a combined multivariate display, standardization is a display transformation based on the
supplied observed responses:

\begin{equation}
z_{ij}=\frac{y_{ij}-\bar y_j}{s_j},\qquad
\hat z_{ij}=\frac{\hat y_{ij}-\bar y_j}{s_j},\qquad
e^{(z)}_{ij}=z_{ij}-\hat z_{ij},
\end{equation}

where $s_j$ is the sample standard deviation calculated with `ddof=1`. Constant response columns
are rejected because their standardized diagnostics are undefined.

The first supported prediction display contains standardized observed-versus-predicted values,
standardized residuals versus standardized predictions, and response-wise standardized RMSE. A
pooled residual histogram and fitted normal density are not part of the initial contract.

Real-data examples may calculate fixed-parameter out-of-fold predictions after the user has chosen
component counts from the same observations. Such results are labeled
`selection-conditioned OOF predictions`; they are descriptive diagnostics, not unbiased nested-CV
or external-test estimates.

### Established PLS analysis contract

The ordinary PLS surface is restricted to public `PLSRegression` quantities:

- X scores;
- X loadings;
- Y loadings;
- regression coefficients;
- raw score-distance and X-reconstruction-residual diagnostics for suitable larger data.

Scores, loadings, and coefficients support explicit component or response subsets. Predictor
loadings and coefficients use the same explicit bar-versus-line rendering contract as Pi-PLS
predictor directions.

The raw score distance uses the fitted training-score center and covariance, with the
Moore--Penrose inverse for numerical rank deficiency. The X reconstruction residual uses the
public transform/inverse-transform round trip. The initial implementation draws no theoretical
limits and performs no automatic observation labeling.

Biplot scaling must be mathematically stated and must preserve the selected score-loading
reconstruction. Biplots are demonstrated only where the number of predictor arrows is readable;
they are not generated for the high-dimensional spectral datasets.

Observation diagnostics initially report raw descriptive quantities only. Theoretical probability
limits, confidence ellipses, VIP scores, automatic variable selection, uncertainty intervals,
permutation tests, and contribution plots require separate decisions and are not implied by this
architecture.

### Plotting and artifact contract

Plotting functions:

- accept computed results and explicit labels rather than retrieving hidden data;
- return the Matplotlib `Figure` and named axes;
- do not call `show()`;
- do not save files;
- do not mutate estimators or input arrays;
- validate dimensions, labels, selected components, and selected responses.

For real-data analyses, numerical CSV files are canonical and PDFs are views reconstructed from
those tables. Prediction tables retain original-unit and standardized observed values, predictions,
residuals, model identity, sample identity, response identity, and prediction provenance. Pi-PLS
response-direction tables retain $q_{jk}$, $d_k$, and $d_kq_{jk}$ separately rather than exporting
only the plotted product. Pickled estimators are not analysis artifacts.

## Implementation sequence

Implementation proceeds as a series of small patches:

1. pure Pi-PLS display-factor and prediction-diagnostic computations — complete;
2. Pi-PLS decomposition and prediction plots — complete;
3. ordinary PLS scores, loadings, and coefficient analysis — complete;
4. example-local fixed-model OOF and post-analysis artifact helpers, integrated first with
   Pulp — complete;
5. Sugarcane spectral line analysis — complete;
6. Tobacco pagination and observation diagnostics — complete;
7. the Pulp biplot and a final cross-dataset analysis-surface review — completed by Decision 0043.

The temporary standalone `09_model_inspection.py` demonstration was removed after the complete
Pulp, Sugarcane, and Tobacco workflows superseded it. The reusable package APIs remain documented
directly and exercised through focused tests and the complete real-data examples.

The package numerical core, estimator fitting, path-selection engine, and current component-path
helpers do not change as part of this decision.

## Consequences

- Component-path plots remain selection diagnostics rather than being repurposed as post-fit
  interpretation.
- Full-data decomposition, score, loading, and coefficient plots are described as fitted-model
  views and do not imply predictive validation.
- Prediction provenance is part of every prediction-diagnostic result and artifact.
- Pi-PLS-specific $P$, $D$, and $Q$ interpretation is supported without conflating rotations with
  ordinary PLS loadings.
- Low-dimensional scalar and high-dimensional spectral examples can share one plotting surface
  while making their rendering choice explicit.
- Optional plotting does not become a core installation requirement.
- Advanced PLS diagnostics remain deferred until their definitions and interpretation boundaries
  are reviewed separately.
