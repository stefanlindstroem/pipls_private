# Pi-PLS theory overview

Pi-PLS is a multivariate linear-regression method with two explicit rank controls. It first retains
a rank-controlled predictor subspace, then constructs a response subspace from cross-covariance,
and finally diagonalizes the reduced regression map into paired predictor-response modes.

## Scientific source and package scope

The scientific source for the construction summarized on this page is the companion manuscript:

> Vishal Agrawal, Fritjof Nilsson, and Stefan B. Lindström, “Panoramic Partial Least Squares
> (Pi-PLS): Transparent, parsimonious, and more interpretable multivariate regression model.”
> Manuscript under revision at *Computers & Chemical Engineering*, CACE-D-26-00847.

The manuscript states the fixed mathematical core for centered predictor and response matrices.
The package implements that core and adds ordinary software facilities around it, including optional
scaling, numerical-rank checks, configurable SVD solvers, cross-validated search, immutable result
records, and prediction diagnostics. Those package capabilities do not change the equations below.
They also do not imply that every package default or practical workflow reproduces a choice made in
the manuscript.

The Pulp source paper documents the provenance and scientific context of the Pulp dataset; it is not
the theoretical reference for Pi-PLS. Pulp provenance and analysis are covered in the
[reference-dataset guide](datasets.md) and [Pulp tutorial](tutorials/pulp.md). Citation metadata for
the software and companion manuscript is maintained on the [citation page](citation.md).

## Problem setting and two rank controls

Let $\mathbf{X}\in\mathbb{R}^{n\times p}$ contain column-centered predictors and let
$\mathbf{Y}\in\mathbb{R}^{n\times q}$ contain column-centered responses. Multivariate linear
regression seeks
$\mathbf{B}\in\mathbb{R}^{p\times q}$ such that

\begin{equation}
\mathbf{Y}=\mathbf{X}\mathbf{B}+\mathbf{E}.
\end{equation}

Pi-PLS separates two structural choices:

- the predictor rank $r_\pi$, which controls the dimension of the retained predictor subspace;
- the component count $h$, which controls the number of paired predictor-response modes.

The admissible dimensions satisfy

\begin{equation}
1\leq h\leq\min(r_\pi,q),
\qquad
r_\pi\leq\min(n,p).
\end{equation}

For a centered estimator fit, the effective sample-space bound becomes
$r_\pi\leq\min(n-1,p)$. The implementation also requires $r_\pi$ not to exceed the verified numerical
rank of the preprocessed predictor matrix.

## Canonical terminology

The package uses the following terms for the fixed Pi-PLS construction:

| Object or public name | Canonical meaning |
|---|---|
| $\mathbf{\Pi}$ | retained predictor basis |
| $\mathbf{\Pi}\mathbf{\Pi}^{\mathsf T}$ | retained-subspace projector |
| $\mathbf{P}$ | orthonormal predictor directions |
| $\mathbf{Q}$ | orthonormal response directions |
| $d_k=D_{kk}$ | dilation of paired latent mode $k$ |
| $\mathbf{X}\mathbf{P}$ | predictor scores |
| $\mathbf{Y}\mathbf{Q}$ | response scores |
| $(P_{:k},d_k,Q_{:k})$ | paired latent mode $k$ |
| `predictor_rank` | retained predictor-subspace dimension $r_\pi$ |
| `n_components` | number of paired latent modes $h$ |

The word **direction** is the primary mathematical term for columns of $\mathbf{P}$ and
$\mathbf{Q}$. Existing Python field names such as
`predictor_rotations`, `response_rotations`, `x_rotations_`, and `y_rotations_` remain public
interface names; they do not make $\mathbf{P}$ or $\mathbf{Q}$ projection matrices. The projectors
onto the final direction spans are $\mathbf{P}\mathbf{P}^{\mathsf T}$ and
$\mathbf{Q}\mathbf{Q}^{\mathsf T}$.

The Pi-PLS directions are also distinct from `x_loadings_` and `y_loadings_`, which are
least-squares reconstruction loadings for the centered or centered-and-scaled training blocks.
For response-side factor displays, the package stores response-by-mode weighted directions
$\mathbf{Q}\mathbf{D}$, where column $k$ is $d_kQ_{:k}$. The manuscript's mode-by-response
orientation is the transpose:

\begin{equation}
\mathbf{D}\mathbf{Q}^{\mathsf T}=(\mathbf{Q}\mathbf{D})^{\mathsf T}.
\end{equation}

The API word “component” is retained because it is familiar in regression software. In Pi-PLS,
`n_components` counts paired latent modes, not predictor-SVD directions and not synthetic latent
components. Likewise, `predictor_rank` is a retained observed-subspace dimension; “predictor signal
rank” is reserved for synthetic settings where the noiseless generating rank is known.

## 1. Rank-controlled predictor projection

Take a singular value decomposition

\begin{equation}
\mathbf{X}=\mathbf{U}_{\mathrm{X}}\mathbf{S}_{\mathrm{X}}\mathbf{V}_{\mathrm{X}}^{\mathsf T}
\end{equation}

and retain the leading $r_\pi$ right singular vectors:

\begin{equation}
\mathbf{\Pi}=\mathbf{V}_{\mathrm{X}}(:,1:r_\pi)\in\mathbb{R}^{p\times r_\pi},
\qquad
\mathbf{\Pi}^{\mathsf T}\mathbf{\Pi}=\mathbf{I}_{r_\pi}.
\end{equation}

The matrix $\mathbf{\Pi}\mathbf{\Pi}^{\mathsf T}$ is the orthogonal projector onto the retained
predictor subspace, so

\begin{equation}
\mathbf{X}=\mathbf{X}\mathbf{\Pi}\mathbf{\Pi}^{\mathsf T}+\mathbf{X}(\mathbf{I}_p-\mathbf{\Pi}\mathbf{\Pi}^{\mathsf T}).
\end{equation}

Define the retained predictor scores

\begin{equation}
\mathbf{Z}=\mathbf{X}\mathbf{\Pi}\in\mathbb{R}^{n\times r_\pi}.
\end{equation}

The first term, $\mathbf{Z}\mathbf{\Pi}^{\mathsf T}$, is the rank-$r_\pi$ predictor approximation.
The second term is orthogonal to the retained subspace and contributes a truncation residual.
Substituting the decomposition into the regression relation gives

\begin{equation}
\mathbf{Y}=\mathbf{Z}\mathbf{\Pi}^{\mathsf T}\mathbf{B}+\mathbf{E}',
\qquad
\mathbf{E}'=\mathbf{E}+\mathbf{X}(\mathbf{I}_p-\mathbf{\Pi}\mathbf{\Pi}^{\mathsf T})\mathbf{B}.
\end{equation}

This step is response-independent. Any predictive direction removed by the choice of $r_\pi$ cannot
be recovered later by increasing $h$.

## 2. Covariance-driven response projection

Within the retained predictor representation, form

\begin{equation}
\boldsymbol{\Sigma}_{\mathrm{ZY}}=\mathbf{Z}^{\mathsf T}\mathbf{Y}\in\mathbb{R}^{r_\pi\times q}.
\end{equation}

For a prescribed $h$, Pi-PLS selects an orthonormal response basis
$\mathbf{C}\in\mathbb{R}^{q\times h}$ by solving

\begin{equation}
\max_{\mathbf{C}^{\mathsf T}\mathbf{C}=\mathbf{I}_h}
\left\|\boldsymbol{\Sigma}_{\mathrm{ZY}}\mathbf{C}\right\|_{\mathrm{F}}^2.
\end{equation}

Because

\begin{equation}
\left\|\boldsymbol{\Sigma}_{\mathrm{ZY}}\mathbf{C}\right\|_{\mathrm{F}}^2
=
\operatorname{tr}\!\left(
\mathbf{C}^{\mathsf T}
\boldsymbol{\Sigma}_{\mathrm{ZY}}^{\mathsf T}
\boldsymbol{\Sigma}_{\mathrm{ZY}}
\mathbf{C}
\right),
\end{equation}

this is an orthonormal trace-maximization problem. If

\begin{equation}
\boldsymbol{\Sigma}_{\mathrm{ZY}}=\mathbf{U}\mathbf{S}\mathbf{V}^{\mathsf T},
\end{equation}

then the leading $h$ right singular vectors span an optimum:

\begin{equation}
\mathbf{C}=\mathbf{V}_{(:,1:h)}.
\end{equation}

Thus $\mathbf{C}$ selects the response subspace with the largest retained squared cross-covariance
with $\mathbf{Z}$. The maximizing subspace is the meaningful object; individual basis vectors are
not unique when the relevant singular values are repeated.

## 3. Least squares in the reduced coordinates

Project the response matrix onto the selected response subspace and fit

\begin{equation}
\mathbf{Y}\mathbf{C}=\mathbf{Z}\mathbf{W}+\mathbf{E}'',
\end{equation}

where $\mathbf{W}\in\mathbb{R}^{r_\pi\times h}$. The minimum-norm least-squares solution is

\begin{equation}
\mathbf{W}=\mathbf{Z}^+\mathbf{Y}\mathbf{C}.
\end{equation}

Before diagonalization, the regression map in the centered coordinates is

\begin{equation}
\mathbf{B}_{\mathrm{cs}}=\mathbf{\Pi}\mathbf{W}\mathbf{C}^{\mathsf T},
\end{equation}

and the fitted responses are

\begin{equation}
\widehat{\mathbf{Y}}=\mathbf{X}\mathbf{B}_{\mathrm{cs}}=\mathbf{Z}\mathbf{W}\mathbf{C}^{\mathsf T}.
\end{equation}

The implementation uses a least-squares solver rather than forming an explicit inverse.

## 4. Diagonal latent coupling {#diagonal-latent-coupling}

Take the economical singular value decomposition

\begin{equation}
\mathbf{W}=\mathbf{M}\mathbf{D}\mathbf{N}^{\mathsf T},
\end{equation}

where $\mathbf{M}\in\mathbb{R}^{r_\pi\times h}$ and
$\mathbf{N}\in\mathbb{R}^{h\times h}$ have orthonormal columns,
and

\begin{equation}
\mathbf{D}=\operatorname{diag}(d_1,\ldots,d_h),
\qquad
d_1\geq d_2\geq\cdots\geq d_h\geq0.
\end{equation}

Define

\begin{equation}
\mathbf{P}=\mathbf{\Pi}\mathbf{M}\in\mathbb{R}^{p\times h},
\qquad
\mathbf{Q}=\mathbf{C}\mathbf{N}\in\mathbb{R}^{q\times h}.
\end{equation}

Then

\begin{equation}
\mathbf{P}^{\mathsf T}\mathbf{P}=\mathbf{I}_h,
\qquad
\mathbf{Q}^{\mathsf T}\mathbf{Q}=\mathbf{I}_h,
\end{equation}

and the reduced regression relation becomes

\begin{equation}
\mathbf{Y}\mathbf{Q}=\mathbf{X}\mathbf{P}\mathbf{D}+\mathbf{E}_\pi,
\qquad
\mathbf{E}_\pi=\mathbf{E}''\mathbf{N}.
\end{equation}

The predictor score vector $\mathbf{X}P_{:k}$ is coupled only to the response score
vector $\mathbf{Y}Q_{:k}$, with dilation $d_k$. This is the one-to-one, mode-wise interpretation
central to Pi-PLS. Orthogonal rotation by $\mathbf{N}$ preserves the Frobenius norm of the
residual and orthogonally transforms its covariance; it preserves covariance
eigenvalues but does not generally leave the covariance matrix itself unchanged.

The equivalent regression-map factorizations are

\begin{equation}
\boxed{
\mathbf{B}_{\mathrm{cs}}
=\mathbf{P}\mathbf{D}\mathbf{Q}^{\mathsf T}
=\mathbf{\Pi}\mathbf{W}\mathbf{C}^{\mathsf T}
}.
\end{equation}

For new centered predictors $\mathbf{X}_{\mathrm{new}}$,

\begin{equation}
\widehat{\mathbf{Y}}_{\mathrm{new}}=\mathbf{X}_{\mathrm{new}}\mathbf{P}\mathbf{D}\mathbf{Q}^{\mathsf T}.
\end{equation}

## Why the method is panoramic

Standard deflation-based PLS algorithms construct successive components while removing previously
modelled predictor variation. Pi-PLS instead fixes one rank-controlled predictor representation
$\mathbf{Z}=\mathbf{X}\mathbf{\Pi}$ and derives all $h$ coupled modes from that undeflated retained
space through closed-form matrix decompositions. The retained predictor space therefore remains
available as a whole during the covariance and regression stages; in this sense, the view is
panoramic.

## Interpretation of $r_\pi$ and $h$ {#interpretation-of-the-ranks}

The two ranks control different forms of complexity.

The predictor rank $r_\pi$ determines which predictor variation remains available to the model. If
$r_\pi$ is too small, response-relevant variation can be discarded irreversibly. Retaining more
predictor directions does not force all of them to contribute strongly, because the response
subspace and latent least-squares map are estimated afterward.

The component count $h$ determines the rank and interpretive complexity of the diagonal
predictor-response coupling. Increasing $h$ adds paired modes, but it cannot recover predictor
directions excluded by $r_\pi$ and cannot exceed the number of responses.

The phrase “predictor signal rank” is appropriate for a synthetic data-generating model when its
noiseless rank is known. For an observed noisy predictor matrix, $\mathbf{\Pi}$ is more accurately
described as a retained predictor basis: leading observed singular directions are not guaranteed to
separate signal from noise exactly.

## Nominal fitted dimension

After $\mathbf{\Pi}$ has been fixed, the fitted representation is described by $\mathbf{M}$,
$\mathbf{D}$, and $\mathbf{Q}$. These contain $r_\pi h$, $h$, and $qh$ entries. The orthonormality
constraints on $\mathbf{M}$ and $\mathbf{Q}$ each remove $h(h+1)/2$ degrees of freedom. The resulting
nominal fitted dimension is

\begin{equation}
\boxed{(r_\pi+q-h)h}.
\end{equation}

This count concerns the fitted representation after the retained predictor basis has been fixed. It
does not treat the $ph$ entries of $\mathbf{P}=\mathbf{\Pi}\mathbf{M}$ as independently free
parameters.

## Relationships to established methods

### Ordinary least squares

If predictor truncation preserves the estimable predictor row space and $h$ retains every estimable
response-side direction, then

\begin{equation}
\widehat{\mathbf{Y}}=\mathbf{Z}\mathbf{W}\mathbf{C}^{\mathsf T}
\end{equation}

coincides with the multivariate OLS fitted response. In that limit, Pi-PLS is an orthogonal latent
reparameterization of the same fitted map.

### Reduced-rank regression

Both Pi-PLS and reduced-rank regression produce low-rank coefficient structures. Reduced-rank
regression obtains the best rank-$h$ approximation of the OLS fit under its least-squares
criterion. Pi-PLS first fixes a rank-controlled predictor representation, selects a response
subspace by cross-covariance, and then diagonalizes the reduced least-squares map into one-to-one
paired modes.

### Canonical correlation analysis

CCA also constructs paired predictor and response variates with a diagonal association structure,
but classical CCA maximizes normalized correlation after within-block whitening. Pi-PLS instead
uses an unwhitened cross-covariance criterion inside the retained predictor representation and then
estimates a predictive least-squares map. Its diagonal relation is analogous to CCA structurally,
not identical to the CCA objective.

### PLS and PLS-SVD

PLS and Pi-PLS both use predictor-response covariance. Standard multicomponent PLS is commonly
constructed through iterative extraction and deflation. Pi-PLS uses a fixed retained predictor
space followed by SVD and least squares.

PLS-SVD derives predictor and response directions directly from a cross-covariance operator.
Pi-PLS first determines $\mathbf{Z}=\mathbf{X}\mathbf{\Pi}$ from predictor singular structure and
uses $\mathbf{Z}^{\mathsf T}\mathbf{Y}$ to select only the response subspace. The final
$\mathbf{P}$ and $\mathbf{Q}$ arise after the least-squares map $\mathbf{W}$ is
estimated and diagonalized; they are not both obtained directly from the first cross-covariance
SVD.

## Package realization

`PiPLSRegression` learns predictor and response means and, when `scale=True`, sample-standard-
deviation scales from the training data. It applies the fixed construction above in the resulting
centered or centered-and-scaled coordinates, then transforms the regression map back to original
units for `coef_`, `intercept_`, and `predict()`.

A fitted estimator exposes the Pi-PLS-specific factorization in `decomposition_`:

- `predictor_rotations`: $\mathbf{P}$;
- `dilation`: $(d_1,\ldots,d_h)$;
- `response_rotations`: $\mathbf{Q}$;
- `standardized_regression_map`: $\mathbf{P}\mathbf{D}\mathbf{Q}^{\mathsf T}$;
- numerical-rank and resolved predictor-SVD diagnostics.

The construction matrices $\mathbf{\Pi}$, $\mathbf{C}$, and $\mathbf{W}$ remain private.
Conventional scores, reconstruction loadings, coefficients, and transformations are exposed
separately from the $\mathbf{P}$, $\mathbf{D}$, and $\mathbf{Q}$ factorization.

Singular-vector signs are arbitrary. Repeated or nearly repeated singular values identify invariant
subspaces rather than unique ordered columns. Numerical validation should therefore compare
regression maps, predictions, projections, principal angles, or singular values rather than raw
basis columns.

## Selection, validation, and synthetic-data boundaries

The equations above define one fixed $(h,r_\pi)$ model. `PiPLSSearchCV` is a package-level facility
for evaluating admissible fixed models under fold-local preprocessing and response-standardized
CV-MSE. Its search policies and the practical real-data workflows documented elsewhere are not
redefined by this theory page and need not duplicate the manuscript’s experimental choices.

The package offers two synthetic-data purposes. `make_pipls_latent_geometry()` implements the
companion manuscript’s Gaussian latent data-generating model directly. The older
`make_pipls_regression()` and `make_pipls_train_test()` functions remain broader configurable
package generators. See the [companion-manuscript synthetic-data guide](manuscript_reproduction.md)
for the exact distribution and reproducibility boundary, and the
[dataset and generator guide](datasets.md) for the broader package distinction.

The [Pulp tutorial’s fixed-rank selection section](tutorials/pulp.md#select-the-fixed-rank-pair)
and [path-selection details](path_analysis.md) define the package’s general selection and
validation contracts.
