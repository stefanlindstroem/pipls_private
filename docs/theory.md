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

Let $X\in\mathbb{R}^{n\times p}$ contain column-centered predictors and let
$Y\in\mathbb{R}^{n\times q}$ contain column-centered responses. Multivariate linear regression seeks
$B\in\mathbb{R}^{p\times q}$ such that

\begin{equation}
Y=XB+E.
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

## 1. Rank-controlled predictor projection

Take a singular value decomposition

\begin{equation}
X=U_XS_XV_X^{\mathsf T}
\end{equation}

and retain the leading $r_\pi$ right singular vectors:

\begin{equation}
\Pi=V_{X(:,1:r_\pi)}\in\mathbb{R}^{p\times r_\pi},
\qquad
\Pi^{\mathsf T}\Pi=I_{r_\pi}.
\end{equation}

The matrix $\Pi\Pi^{\mathsf T}$ is the orthogonal projector onto the retained predictor subspace, so

\begin{equation}
X=X\Pi\Pi^{\mathsf T}+X(I_p-\Pi\Pi^{\mathsf T}).
\end{equation}

Define the retained predictor scores

\begin{equation}
Z=X\Pi\in\mathbb{R}^{n\times r_\pi}.
\end{equation}

The first term, $Z\Pi^{\mathsf T}$, is the rank-$r_\pi$ predictor approximation. The second term is
orthogonal to the retained subspace and contributes a truncation residual. Substituting the
decomposition into the regression relation gives

\begin{equation}
Y=Z\Pi^{\mathsf T}B+E',
\qquad
E'=E+X(I_p-\Pi\Pi^{\mathsf T})B.
\end{equation}

This step is response-independent. Any predictive direction removed by the choice of $r_\pi$ cannot
be recovered later by increasing $h$.

## 2. Covariance-driven response projection

Within the retained predictor representation, form

\begin{equation}
\Sigma_{ZY}=Z^{\mathsf T}Y\in\mathbb{R}^{r_\pi\times q}.
\end{equation}

For a prescribed $h$, Pi-PLS selects an orthonormal response basis
$C\in\mathbb{R}^{q\times h}$ by solving

\begin{equation}
\max_{C^{\mathsf T}C=I_h}
\left\|\Sigma_{ZY}C\right\|_F^2.
\end{equation}

Because

\begin{equation}
\left\|\Sigma_{ZY}C\right\|_F^2
=
\operatorname{tr}\!\left(C^{\mathsf T}\Sigma_{ZY}^{\mathsf T}\Sigma_{ZY}C\right),
\end{equation}

this is an orthonormal trace-maximization problem. If

\begin{equation}
\Sigma_{ZY}=USV^{\mathsf T},
\end{equation}

then the leading $h$ right singular vectors span an optimum:

\begin{equation}
C=V_{(:,1:h)}.
\end{equation}

Thus $C$ selects the response subspace with the largest retained squared cross-covariance with $Z$.
The maximizing subspace is the meaningful object; individual basis vectors are not unique when the
relevant singular values are repeated.

## 3. Least squares in the reduced coordinates

Project the response matrix onto the selected response subspace and fit

\begin{equation}
YC=ZW+E'',
\end{equation}

where $W\in\mathbb{R}^{r_\pi\times h}$. The minimum-norm least-squares solution is

\begin{equation}
W=Z^+YC.
\end{equation}

Before diagonalization, the regression map in the centered coordinates is

\begin{equation}
B_{\mathrm{cs}}=\Pi WC^{\mathsf T},
\end{equation}

and the fitted responses are

\begin{equation}
\widehat Y=XB_{\mathrm{cs}}=ZWC^{\mathsf T}.
\end{equation}

The implementation uses a least-squares solver rather than forming an explicit inverse.

## 4. Diagonal latent coupling {#diagonal-latent-coupling}

Take the economical singular value decomposition

\begin{equation}
W=MDN^{\mathsf T},
\end{equation}

where $M\in\mathbb{R}^{r_\pi\times h}$ and $N\in\mathbb{R}^{h\times h}$ have orthonormal columns,
and

\begin{equation}
D=\operatorname{diag}(d_1,\ldots,d_h),
\qquad
d_1\geq d_2\geq\cdots\geq d_h\geq0.
\end{equation}

Define

\begin{equation}
P=\Pi M\in\mathbb{R}^{p\times h},
\qquad
Q=CN\in\mathbb{R}^{q\times h}.
\end{equation}

Then

\begin{equation}
P^{\mathsf T}P=I_h,
\qquad
Q^{\mathsf T}Q=I_h,
\end{equation}

and the reduced regression relation becomes

\begin{equation}
YQ=XPD+E_\pi,
\qquad
E_\pi=E''N.
\end{equation}

Column $k$ of $XP$ is coupled only to column $k$ of $YQ$, with dilation $d_k$. This is the
one-to-one, mode-wise interpretation central to Pi-PLS. Orthogonal rotation by $N$ preserves the
Frobenius norm of the residual and orthogonally transforms its covariance; it preserves covariance
eigenvalues but does not generally leave the covariance matrix itself unchanged.

The equivalent regression-map factorizations are

\begin{equation}
\boxed{
B_{\mathrm{cs}}
=PDQ^{\mathsf T}
=\Pi WC^{\mathsf T}
}.
\end{equation}

For new centered predictors $X_{\mathrm{new}}$,

\begin{equation}
\widehat Y_{\mathrm{new}}=X_{\mathrm{new}}PDQ^{\mathsf T}.
\end{equation}

## Why the method is panoramic

Standard deflation-based PLS algorithms construct successive components while removing previously
modelled predictor variation. Pi-PLS instead fixes one rank-controlled predictor representation
$Z=X\Pi$ and derives all $h$ coupled modes from that undeflated retained space through closed-form
matrix decompositions. The retained predictor space therefore remains available as a whole during
the covariance and regression stages; in this sense, the view is panoramic.

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
noiseless rank is known. For an observed noisy predictor matrix, $\Pi$ is more accurately described
as a retained predictor basis: leading observed singular directions are not guaranteed to separate
signal from noise exactly.

## Nominal fitted dimension

After $\Pi$ has been fixed, the fitted representation is described by $M$, $D$, and $Q$. These
contain $r_\pi h$, $h$, and $qh$ entries. The orthonormality constraints on $M$ and $Q$ each remove
$h(h+1)/2$ degrees of freedom. The resulting nominal fitted dimension is

\begin{equation}
\boxed{(r_\pi+q-h)h}.
\end{equation}

This count concerns the fitted representation after the retained predictor basis has been fixed. It
does not treat the $ph$ entries of $P=\Pi M$ as independently free parameters.

## Relationships to established methods

### Ordinary least squares

If predictor truncation preserves the estimable predictor row space and $h$ retains every estimable
response-side direction, then

\begin{equation}
\widehat Y=ZWC^{\mathsf T}
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
Pi-PLS first determines $Z=X\Pi$ from predictor singular structure and uses $Z^{\mathsf T}Y$ to
select only the response subspace. The final $P$ and $Q$ arise after the least-squares map $W$ is
estimated and diagonalized; they are not both obtained directly from the first cross-covariance
SVD.

## Package realization

`PiPLSRegression` learns predictor and response means and, when `scale=True`, sample-standard-
deviation scales from the training data. It applies the fixed construction above in the resulting
centered or centered-and-scaled coordinates, then transforms the regression map back to original
units for `coef_`, `intercept_`, and `predict()`.

A fitted estimator exposes the Pi-PLS-specific factorization in `decomposition_`:

- `predictor_rotations`: $P$;
- `dilation`: $(d_1,\ldots,d_h)$;
- `response_rotations`: $Q$;
- `standardized_regression_map`: $PDQ^{\mathsf T}$;
- numerical-rank and resolved predictor-SVD diagnostics.

The construction matrices $\Pi$, $C$, and $W$ remain private. Conventional scores, reconstruction
loadings, coefficients, and transformations are exposed separately from the $P$, $D$, and $Q$
factorization.

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
package generators. See the [dataset and generator guide](datasets.md) for the exact distinction.

The [Pulp tutorial’s fixed-rank selection section](tutorials/pulp.md#select-the-fixed-rank-pair)
and [path-selection details](path_analysis.md) define the package’s general selection and
validation contracts.
