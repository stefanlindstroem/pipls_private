# Pi-PLS theory overview

Pi-PLS is a multivariate linear-regression method with two explicit rank controls. It first retains
a predictor signal subspace and then couples predictor and response modes through a diagonal latent
map.

## Problem setting

Let $X\in\mathbb{R}^{n\times p}$ contain predictors and let
$Y\in\mathbb{R}^{n\times q}$ contain one or more responses. `PiPLSRegression` centers both blocks
and, when `scale=True`, divides each column by its training-sample standard deviation. Write the
resulting matrices as $X_{\mathrm{cs}}$ and $Y_{\mathrm{cs}}$.

The fixed model is controlled by:

- predictor rank $r_\pi$, which determines how much of the predictor variation is retained;
- component count $h$, which determines the number of paired predictor-response modes.

The admissible dimensions satisfy $1\leq h\leq\min(r_\pi,q)$. Because centering removes one sample
direction, a direct estimator fit also requires $r_\pi\leq\min(n-1,p)$.

## Predictor signal subspace

Compute a thin singular value decomposition of the preprocessed predictor matrix and retain the
leading $r_\pi$ right singular vectors in
$\Pi\in\mathbb{R}^{p\times r_\pi}$. The reduced predictor coordinates are

\begin{equation}
Z=X_{\mathrm{cs}}\Pi.
\end{equation}

This step separates predictor-side rank control from the later response-coupled component count.
Variation discarded by $\Pi$ cannot enter the fitted regression map.

## Response subspace

The cross-product $Z^{\mathsf T}Y_{\mathrm{cs}}$ identifies response directions associated with
the retained predictor subspace. Let $C\in\mathbb{R}^{q\times h}$ contain its leading $h$ right
singular vectors.

The reduced response coordinates are $Y_{\mathrm{cs}}C$. Pi-PLS solves the least-squares problem

\begin{equation}
ZW\approx Y_{\mathrm{cs}}C,
\end{equation}

where $W\in\mathbb{R}^{r_\pi\times h}$.

## Diagonal latent coupling

Factor the reduced map as

\begin{equation}
W=MDN^{\mathsf T},
\end{equation}

where $M$ and $N$ have orthonormal columns and $D$ is diagonal with nonnegative entries. Define

\begin{equation}
P=\Pi M,
\qquad
Q=CN.
\end{equation}

The centered and scaled regression map is then

\begin{equation}
B_{\mathrm{cs}}=PDQ^{\mathsf T}=\Pi WC^{\mathsf T}.
\end{equation}

The columns of $P$ and $Q$ are paired predictor and response rotations. The diagonal entry $d_k$
scales the contribution of pair $k$. Predictions in preprocessed coordinates are

\begin{equation}
\widehat{Y}_{\mathrm{cs}}=X_{\mathrm{cs}}PDQ^{\mathsf T}.
\end{equation}

`PiPLSRegression` transforms this map back to the original predictor and response units when it
constructs `coef_`, `intercept_`, and `predict()` output.

## Interpretation of the ranks

The predictor rank $r_\pi$ controls the dimension of the predictor signal space available to the
regression. If it is too small, useful predictor variation may be discarded. If it is unnecessarily
large, weak or nuisance predictor directions may enter the reduced regression problem.

The component count $h$ controls the number of diagonal predictor-response pairs. Increasing $h$
adds paired modes but cannot exceed either the retained predictor rank or the number of responses.

The two ranks therefore describe different forms of complexity. `PiPLSPathCV` evaluates predictor
rank conditionally for each requested component count and reports one path row per component count.
The user then fits a separate fixed model with both selected values stated explicitly.

## Fitted quantities

A fitted estimator exposes two related analysis surfaces.

Pi-PLS-specific quantities are stored in `decomposition_` with descriptive field names:

- `predictor_rotations`: $P$;
- `dilation`: the nonnegative diagonal values of $D$;
- `response_rotations`: $Q$;
- `standardized_regression_map`: $PDQ^{\mathsf T}$.

The intermediate construction matrices $\Pi$, $C$, and $W$ remain private. They define the method
above but are not required for fitted-model interpretation or downstream prediction.

The estimator also exposes conventional PLS-family quantities:

- predictor and response scores;
- predictor and response loadings;
- regression coefficients;
- forward and inverse latent transformations.

The shared quantities support ordinary score, loading, coefficient, biplot, and observation
diagnostics. The $P$, $D$, and $Q$ factorization provides the additional Pi-PLS-specific view.

## Relation to other low-rank regressions

Pi-PLS belongs to the PLS family because it constructs predictor and response latent variables from
cross-block information and provides the usual score and loading representations. Its distinctive
feature is the separate predictor truncation rank followed by the diagonal coupling
$PDQ^{\mathsf T}$.

When the retained predictor subspace spans all centered predictor variation, the predictor
truncation no longer removes directions. When the component count includes every admissible paired
mode, the diagonal factorization represents the complete reduced least-squares map. These limits
connect Pi-PLS to familiar reduced-rank and least-squares constructions without changing the
implemented two-rank parameterization.

## Identifiability and numerical rank

Singular-vector signs are arbitrary. Repeated or nearly repeated singular values identify invariant
subspaces rather than unique ordered basis vectors. Consequently, numerical validation should
compare regression maps, predictions, projection matrices, principal angles, or singular values
rather than requiring raw basis columns to have fixed signs.

The implementation checks the retained predictor singular values against a scale-dependent
floating-point tolerance. A requested predictor rank must not exceed the verified numerical rank of
the preprocessed predictor matrix.

## Model selection and validation

Centering and scaling are learned separately inside every training fold used by `PiPLSPathCV`.
Response-standardized CV-MSE gives each response equal weight after scaling by its training-fold
standard deviation. The component-path fold SD describes variation across the realized folds; it is
not a confidence interval.

A path is a model-selection diagnostic. After inspecting it, fit a fixed `PiPLSRegression` with the
chosen component count and the predictor rank reported for that row. Prediction diagnostics must
state whether they use fitted values, fixed-parameter out-of-fold predictions, selection-conditioned
out-of-fold predictions, or an independent test set.

The [Pulp tutorial’s fixed-rank selection section](tutorials/pulp.md#select-the-fixed-rank-pair),
[path-selection details](path_analysis.md) define the general selection and validation contracts.

## Reference and scope

The equations on this page describe the method implemented by this package. The worked Pulp
application and dataset provenance are documented in:

Stefan B. Lindström, Rita Ferritsius, Johan E. Carlson, Johan Persson, and Fritjof Nilsson,
“Predicting handsheet properties and enhancing refiner control using fiber analyzer data and latent
variable modeling,” *Computers & Chemical Engineering* **199** (2025), 109143,
[doi:10.1016/j.compchemeng.2025.109143](https://doi.org/10.1016/j.compchemeng.2025.109143).

The [Pulp tutorial](tutorials/pulp.md) connects the implemented construction to that repository
analysis. Broader PLS and reduced-rank context is summarized only to position the implemented method;
this page does not define APIs beyond the package behavior documented and tested in this repository.
